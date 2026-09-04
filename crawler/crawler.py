import asyncio
import concurrent.futures
import threading
from tqdm import tqdm
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin, urlparse

import requests
import yaml

from .archive_crawler import archive_document_to_row, identify_file_type
from .deduplicate import Deduplicator, content_hash
from .extractors.generic import GenericExtractor
from .fetcher import Fetcher
from .frontier import is_candidate_url, normalize_url, prioritize_article_urls
from .language import calculate_language_stats
from .normalize import clean_text
from .parser import detect_page_features, extract_internal_links, likely_category_links, parse_feed, parse_sitemap
from .robots import RobotsCache
from .storage import JsonlCorpusWriter, StateStore


USER_AGENT = "NepaliCorpusResearchBot/1.0 (+research; respectful; contact: local-assignment)"


SITE_CONFIGS: Dict[str, Dict[str, object]] = {
    "onlinekhabar": {"article_hints": [r"/\d{4}/\d{2}/\d{2}/", r"/content/\d+"]},
    "ratopati": {"article_hints": [r"/story/\d+", r"/\d{4}/\d{2}/\d{2}/"]},
    "setopati": {"article_hints": [r"/\w+/\w+/\d+", r"/\d+"]},
    "ekantipur": {"article_hints": [r"/\d{4}/\d{1,2}/\d{1,2}/", r"/news/\d+"]},
    "dcnepal": {"article_hints": [r"/\d{4}/\d{2}/\d{2}/"]},
    "imagekhabar": {"article_hints": [r"/news/\d+", r"/\d{4}/"]},
    "ujyaaloonline": {"article_hints": [r"/story/\d+", r"/\d{4}/"]},
    "nepaliarchives": {"pipeline": "archive"},
    "narchives_repository": {"pipeline": "archive"},
    "nnl_archive": {"pipeline": "archive"},
}


def load_config(path: str = "nepali/configs/sources.yaml") -> Dict[str, object]:
    with open(path, encoding="utf-8") as fp:
        return yaml.safe_load(fp)


class NepaliCrawler:
    def __init__(
        self,
        config: Dict[str, object],
        db_path: str = "crawl_state.db",
        output_path: str = "nepali/data/raw/clean/nepali_corpus.jsonl",
        raw_html_dir: str = "nepali/data/raw/raw_html",
        user_agent: str = USER_AGENT,
    ):
        self.config = config
        self.sources = [s for s in config.get("sources", []) if s.get("enabled", True)]
        self.store = StateStore(db_path)
        self.output_path = output_path
        self.raw_html_dir = Path(raw_html_dir)
        self.raw_html_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent
        self.robots = RobotsCache(user_agent=user_agent, fallback_allow=False)
        self.extractor = GenericExtractor()
        self.dedup = Deduplicator()
        self.counters = {"accepted": 0, "rejected": 0, "duplicates": 0, "near_duplicates": 0, "fetched": 0}
        self._lock = threading.Lock()  # protects counters + dedup across concurrent coroutines

    def discover(self, source: Dict[str, object], max_sitemap_urls: int = 5000) -> int:
        base_url = str(source["base_url"])
        domain = urlparse(base_url).netloc
        robots = self.robots.get_info(base_url)
        discovery = source.get("discovery", {}) or {}
        sitemap_queue = list(dict.fromkeys([*robots.sitemaps, *discovery.get("sitemap_urls", [])]))
        rss_urls = list(discovery.get("rss_urls", []))
        discovered: List[str] = []

        for sitemap_url in tqdm(sitemap_queue[:50], desc=f"[{source['name']}] Sitemaps", unit="sitemap", leave=False, miniters=100):
            try:
                resp = requests.get(sitemap_url, headers={"User-Agent": self.user_agent}, timeout=30)
                if not resp.ok:
                    continue
                urls, nested = parse_sitemap(resp.content)
                discovered.extend(urls)
                for nested_url in nested[:100]:
                    if nested_url not in sitemap_queue:
                        sitemap_queue.append(nested_url)
                if len(discovered) >= max_sitemap_urls:
                    break
            except Exception:
                continue

        for feed_url in rss_urls:
            try:
                resp = requests.get(feed_url, headers={"User-Agent": self.user_agent}, timeout=20)
                if resp.ok:
                    discovered.extend(parse_feed(resp.content))
            except Exception:
                continue

        # Always seed from homepage and category navigation
        try:
            resp = requests.get(base_url, headers={"User-Agent": self.user_agent}, timeout=20)
            if resp.ok:
                discovered.extend(likely_category_links(resp.text, base_url, domain))
                discovered.extend(extract_internal_links(resp.text, base_url, domain))
        except Exception:
            pass

        rows = []
        for url in prioritize_article_urls(discovered):
            normalized = normalize_url(url)
            if is_candidate_url(normalized) and urlparse(normalized).netloc == domain:
                rows.append({"url": url, "normalized_url": normalized, "domain": domain, "source": str(source["name"])})
        return self.store.add_urls(rows)

    async def crawl(self, max_documents: Optional[int] = None, per_source_limit: int = 1000, source_name: Optional[str] = None) -> None:
        selected = [s for s in self.sources if not source_name or s["name"] == source_name]
        loop = asyncio.get_event_loop()
        # Run all blocking discover() calls in parallel using a thread pool
        n_workers = min(len(selected), 16)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="discover") as executor:
            discover_tasks = [
                loop.run_in_executor(executor, self.discover, source)
                for source in selected
            ]
            counts = await asyncio.gather(*discover_tasks)
            total_discovered = sum(counts)
        print(f"\n[crawler] Discovered {total_discovered} URLs across {len(selected)} sources — starting parallel crawl...")
        async with Fetcher(self.user_agent) as fetcher:
            with JsonlCorpusWriter(self.output_path) as writer:
                # Crawl all sources concurrently — each source respects its own rate limit
                await asyncio.gather(*(
                    self._crawl_source(source, fetcher, writer, max_documents, per_source_limit)
                    for source in selected
                ))
        self.store.close()

    async def _crawl_source(self, source: Dict[str, object], fetcher: Fetcher, writer: JsonlCorpusWriter, max_documents: Optional[int], per_source_limit: int) -> None:
        base_url = str(source["base_url"])
        domain = urlparse(base_url).netloc
        crawl_conf = source.get("crawl", {}) or {}
        rate = float(crawl_conf.get("requests_per_second", 0.2))
        delay = self.robots.crawl_delay(base_url)
        if delay:
            rate = min(rate, 1.0 / delay)
        
        sem = asyncio.Semaphore(int(crawl_conf.get("concurrency", 2)))
        source_processed = 0

        pbar = tqdm(
            desc=f"[{source['name']}]",
            unit="url",
            dynamic_ncols=True,
            postfix={"accepted": 0, "rejected": 0, "dupes": 0},
            miniters=100,
        )

        async def one(url: str) -> None:
            if max_documents and self.counters["accepted"] >= max_documents:
                return
            async with sem:
                await self._process_url(source, url, fetcher, writer, rate, int(crawl_conf.get("max_retries", 3)))
            pbar.update(1)
            pbar.set_postfix(
                accepted=self.counters["accepted"],
                rejected=self.counters["rejected"],
                dupes=self.counters["duplicates"] + self.counters["near_duplicates"],
            )

        start_accepted = self.counters["accepted"]
        while (self.counters["accepted"] - start_accepted) < per_source_limit:
            if max_documents and self.counters["accepted"] >= max_documents:
                break
            batch_size = min(50, per_source_limit - (self.counters["accepted"] - start_accepted))
            if max_documents:
                remaining = max_documents - self.counters["accepted"]
                if remaining <= 0:
                    break
                batch_size = min(batch_size, remaining)

            urls = self.store.queued(domain, batch_size)
            if not urls:
                break

            await asyncio.gather(*(one(url) for url in urls))

        pbar.close()

    async def _process_url(self, source: Dict[str, object], url: str, fetcher: Fetcher, writer: JsonlCorpusWriter, rate: float, max_retries: int) -> None:
        if not self.robots.can_fetch(url):
            self.store.mark(url, "REJECTED", rejection_reason="robots_disallowed")
            with self._lock:
                self.counters["rejected"] += 1
            return
        self.store.mark_attempt(url)
        result = await fetcher.fetch(url, rate, max_retries=max_retries)
        if result is None:
            self.store.mark(url, "FAILED", rejection_reason="download_failed")
            return
        with self._lock:
            self.counters["fetched"] += 1
        if result.status != 200:
            self.store.mark(url, "FAILED", http_status=result.status, rejection_reason=f"http_{result.status}")
            return
        if "html" not in result.content_type.lower() and source.get("kind") != "archive":
            self.store.mark(url, "REJECTED", http_status=result.status, rejection_reason="non_html")
            return
        html = result.text
        if source.get("kind") == "archive" and "html" not in result.content_type.lower():
            file_type = identify_file_type(url, result.content_type)
            self.store.mark(url, "REJECTED", http_status=result.status, rejection_reason=f"archive_{file_type}_downloaded_needs_offline_extraction")
            return

        # Continuous discovery: extract and queue new internal links from this page
        domain = urlparse(str(source["base_url"])).netloc
        try:
            discovered_links = extract_internal_links(html, url, domain)
            new_rows = []
            for lk in discovered_links:
                norm_lk = normalize_url(lk)
                if is_candidate_url(norm_lk) and urlparse(norm_lk).netloc == domain:
                    new_rows.append({"url": lk, "normalized_url": norm_lk, "domain": domain, "source": str(source["name"])})
            if new_rows:
                self.store.add_urls(new_rows)
        except Exception:
            pass

        doc = self.extractor.extract(url, html, str(source["name"]))
        if doc is None:
            self.store.mark(url, "REJECTED", http_status=result.status, rejection_reason="extraction_failed")
            with self._lock:
                self.counters["rejected"] += 1
            return
        raw_text = doc.raw_text
        doc.text = clean_text(doc.text)
        lang = source.get("language", {}) or {}
        stats = calculate_language_stats(doc.text)
        if len(doc.text) < int(lang.get("min_characters", 450)):
            self.store.mark(url, "REJECTED", http_status=result.status, rejection_reason="too_short")
            with self._lock:
                self.counters["rejected"] += 1
            return
        if stats.devanagari_ratio < float(lang.get("min_devanagari_ratio", 0.62)) or stats.nepali_unicode_ratio < float(lang.get("min_nepali_unicode_ratio", 0.58)):
            self.store.mark(url, "REJECTED", http_status=result.status, rejection_reason="non_nepali")
            with self._lock:
                self.counters["rejected"] += 1
            return

        # Dedup check + write must be atomic to avoid duplicates across concurrent sources
        with self._lock:
            if self.dedup.seen_exact(doc.text):
                self.store.mark(url, "REJECTED", http_status=result.status, rejection_reason="duplicate")
                self.counters["duplicates"] += 1
                return
            if self.dedup.seen_near(doc.text):
                self.store.mark(url, "REJECTED", http_status=result.status, rejection_reason="near_duplicate")
                self.counters["near_duplicates"] += 1
                return
            digest = content_hash(doc.text)
            row = asdict(doc)
            row.update(stats.to_dict())
            row.update(
                {
                    "id": digest,
                    "content_hash": digest,
                    "raw_text": raw_text,
                    "raw_char_count": len(raw_text),
                    "clean_char_count": len(doc.text),
                    "collection_method": "self_crawled",
                }
            )
            if source.get("kind") == "archive":
                row = archive_document_to_row(doc, "html")
                row.update(stats.to_dict(), id=digest, content_hash=digest, raw_char_count=len(raw_text), clean_char_count=len(doc.text))
            writer.write(row)
            self.store.add_document(digest, url, str(source["name"]), len(doc.text), stats.language_score)
            self.store.mark(url, "ACCEPTED", http_status=result.status, content_hash=digest)
            self.counters["accepted"] += 1


def enabled_sources(config: Dict[str, object]) -> Iterable[Dict[str, object]]:
    return (s for s in config.get("sources", []) if s.get("enabled", True))


def summarize_config(config: Dict[str, object]) -> str:
    return "\n".join(f"{s['name']}: {s['base_url']} ({s.get('kind', 'news')})" for s in enabled_sources(config))

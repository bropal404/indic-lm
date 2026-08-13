import argparse
import asyncio
import json
import resource
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

# Raise soft file descriptor limit to hard limit
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(hard, 65535) if hard != resource.RLIM_INFINITY else 65535
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
except Exception:
    pass

import requests

from .crawler import NepaliCrawler, load_config, summarize_config
from .extractors.generic import GenericExtractor
from .frontier import normalize_url, prioritize_article_urls
from .language import calculate_language_stats
from .normalize import clean_text, repair_mojibake
from .parser import detect_page_features, extract_internal_links, likely_category_links, parse_feed, parse_sitemap
from .robots import RobotsCache
from .statistics import CorpusStatistics


USER_AGENT = "NepaliCorpusResearchBot/1.0 (+research; respectful; contact: local-assignment)"


def _get(url: str) -> requests.Response:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=25)
    if not resp.encoding or resp.encoding.lower() in {"iso-8859-1", "latin-1", "windows-1252"}:
        resp.encoding = resp.apparent_encoding or "utf-8"
    resp._content = repair_mojibake(resp.text).encode("utf-8")
    resp.encoding = "utf-8"
    return resp


def run_recon(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    robots_cache = RobotsCache(USER_AGENT, fallback_allow=False)
    extractor = GenericExtractor()
    report: List[Dict[str, object]] = []
    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for source in config["sources"]:
        if args.source and source["name"] != args.source:
            continue
        base = source["base_url"]
        domain = urlparse(base).netloc
        robots = robots_cache.get_info(base)
        discovery = source.get("discovery", {}) or {}
        sitemap_urls = list(dict.fromkeys([*robots.sitemaps, *discovery.get("sitemap_urls", [])]))
        rss_urls = list(discovery.get("rss_urls", []))
        found = []
        nested_sitemaps = []
        feeds = []
        features = {}
        sample_outputs = []
        try:
            home = _get(base)
            if home.ok:
                features = detect_page_features(home.text)
                found.extend(likely_category_links(home.text, base, domain))
                found.extend(extract_internal_links(home.text, base, domain)[:200])
        except Exception as exc:
            features = {"home_error": str(exc)}
        for sitemap in sitemap_urls[:8]:
            try:
                resp = _get(sitemap)
                if resp.ok:
                    urls, nested = parse_sitemap(resp.content)
                    found.extend(urls[: args.sample_urls])
                    nested_sitemaps.extend(nested[:20])
            except Exception:
                pass
        for feed in rss_urls:
            try:
                resp = _get(feed)
                if resp.ok:
                    links = parse_feed(resp.content)
                    feeds.append({"url": feed, "entries": len(links)})
                    found.extend(links)
            except Exception:
                pass
        candidates = prioritize_article_urls(found)[: args.sample_urls]
        duplicates = len(candidates) - len(set(candidates))
        for candidate in candidates[: args.extract_samples]:
            if not robots_cache.can_fetch(candidate):
                continue
            try:
                resp = _get(candidate)
                if not resp.ok or "html" not in resp.headers.get("Content-Type", "").lower():
                    continue
                doc = extractor.extract(candidate, resp.text, source["name"])
                if not doc:
                    continue
                text = clean_text(doc.text)
                stats = calculate_language_stats(text)
                sample_outputs.append(
                    {
                        "url": normalize_url(candidate),
                        "title": doc.title,
                        "first_500": text[:500],
                        "devanagari_ratio": round(stats.devanagari_ratio, 4),
                        "latin_ratio": round(stats.latin_ratio, 4),
                        "language_score": round(stats.language_score, 4),
                    }
                )
            except Exception as exc:
                sample_outputs.append({"url": candidate, "error": str(exc)})
        row = {
            "source": source["name"],
            "base_url": base,
            "kind": source.get("kind", "news"),
            "robots": robots.__dict__,
            "sitemaps": sitemap_urls,
            "nested_sitemap_examples": nested_sitemaps,
            "rss": feeds,
            "home_features": features,
            "candidate_count": len(candidates),
            "duplicate_rate": round(duplicates / len(candidates), 4) if candidates else 0.0,
            "candidate_examples": candidates[:20],
            "extraction_samples": sample_outputs,
            "large_crawl_enabled": False,
        }
        report.append(row)
        print(f"\n## {source['name']}")
        print(f"robots: fetched={robots.fetched} status={robots.status_code} disallow={robots.disallow_count} crawl_delay={robots.crawl_delay}")
        print(f"sitemaps: {len(sitemap_urls)} rss: {len(rss_urls)} candidates: {len(candidates)} duplicate_rate: {row['duplicate_rate']}")
        for sample in sample_outputs[:3]:
            print(f"- {sample.get('title')}")
            print(str(sample.get("first_500", sample.get("error", "")))[:500])
            print(f"  devanagari={sample.get('devanagari_ratio')} score={sample.get('language_score')}")

    json_path = out_dir / "reconnaissance.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_lines = ["# Nepali Corpus Crawler Reconnaissance", ""]
    for row in report:
        md_lines.extend(
            [
                f"## {row['source']}",
                f"- Base URL: {row['base_url']}",
                f"- Robots fetched/status/disallow: {row['robots']['fetched']} / {row['robots']['status_code']} / {row['robots']['disallow_count']}",
                f"- Sitemap examples: {', '.join(row['sitemaps'][:5]) or 'none found'}",
                f"- Candidate URLs sampled: {row['candidate_count']}",
                f"- Duplicate rate in sample: {row['duplicate_rate']}",
                f"- Large crawl enabled: {row['large_crawl_enabled']}",
                "",
            ]
        )
        for sample in row["extraction_samples"][:5]:
            md_lines.extend([f"### {sample.get('title', sample.get('url'))}", f"URL: {sample.get('url')}", f"Devanagari ratio: {sample.get('devanagari_ratio')}", "", str(sample.get("first_500", sample.get("error", "")))[:500], ""])
    (out_dir / "reconnaissance.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nSaved reconnaissance reports to {out_dir}")


def run_crawl(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    crawler = NepaliCrawler(config, db_path=args.db, output_path=args.output, raw_html_dir=args.raw_html_dir)
    asyncio.run(crawler.crawl(max_documents=args.max_documents, per_source_limit=args.per_source_limit, source_name=args.source))
    print(json.dumps(crawler.counters, indent=2))


def run_stats(args: argparse.Namespace) -> None:
    stats = CorpusStatistics()
    stats.load_jsonl(args.path)
    print(stats.report())


def main() -> None:
    parser = argparse.ArgumentParser(description="Nepali web corpus crawler")
    parser.add_argument("--config", default="nepali/configs/sources.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("sources", help="List configured sources")

    recon = sub.add_parser("recon", help="Inspect robots, sitemaps, feeds, URL patterns, and extraction samples")
    recon.add_argument("--source")
    recon.add_argument("--sample-urls", type=int, default=50)
    recon.add_argument("--extract-samples", type=int, default=5)
    recon.add_argument("--report-dir", default="reports")

    crawl = sub.add_parser("crawl", help="Run a resumable respectful crawl")
    crawl.add_argument("--source")
    crawl.add_argument("--max-documents", type=int, default=None, help="Max total documents to collect (default: unlimited)")
    crawl.add_argument("--per-source-limit", type=int, default=50000, help="Max documents per source (default: 50000)")
    crawl.add_argument("--db", default="crawl_state.db")
    crawl.add_argument("--output", default="nepali/data/raw/clean/nepali_corpus.jsonl")
    crawl.add_argument("--raw-html-dir", default="nepali/data/raw/raw_html")

    stats = sub.add_parser("stats", help="Print corpus stats")
    stats.add_argument("path")

    args = parser.parse_args()
    if args.command == "sources":
        print(summarize_config(load_config(args.config)))
    elif args.command == "recon":
        run_recon(args)
    elif args.command == "crawl":
        run_crawl(args)
    elif args.command == "stats":
        run_stats(args)


if __name__ == "__main__":
    main()

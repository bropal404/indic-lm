import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
from urllib.parse import urljoin

import trafilatura
from bs4 import BeautifulSoup


@dataclass
class Document:
    source: str
    url: str
    canonical_url: str
    title: str
    text: str
    raw_text: str
    author: Optional[str] = None
    published_at: Optional[str] = None
    modified_at: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    subtitle: Optional[str] = None
    crawl_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    collection_method: str = "self_crawled"
    extraction_method: str = "generic"
    extra: Dict[str, object] = field(default_factory=dict)


BOILERPLATE_SELECTORS = [
    "script",
    "style",
    "noscript",
    "header",
    "footer",
    "nav",
    "aside",
    "form",
    ".advertisement",
    ".ads",
    ".related",
    ".comments",
    ".comment",
    ".newsletter",
    ".social",
]


def _json_ld_objects(soup: BeautifulSoup) -> List[Dict[str, object]]:
    objects: List[Dict[str, object]] = []
    for node in soup.find_all("script", attrs={"type": re.compile("ld\\+json", re.I)}):
        try:
            data = json.loads(node.string or "")
        except json.JSONDecodeError:
            continue
        queue = data if isinstance(data, list) else [data]
        while queue:
            item = queue.pop(0)
            if isinstance(item, dict):
                objects.append(item)
                graph = item.get("@graph")
                if isinstance(graph, list):
                    queue.extend(graph)
    return objects


def _first_meta(soup: BeautifulSoup, *names: str) -> Optional[str]:
    for name in names:
        node = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
        if node and node.get("content"):
            return node["content"].strip()
    return None


class GenericExtractor:
    def extract(self, url: str, html: str, source: str) -> Optional[Document]:
        soup = BeautifulSoup(html, "lxml")
        canonical = soup.find("link", rel=re.compile("canonical", re.I))
        canonical_url = urljoin(url, canonical["href"].strip()) if canonical and canonical.get("href") else url
        metadata = {}
        for item in _json_ld_objects(soup):
            types = item.get("@type", [])
            if isinstance(types, str):
                types = [types]
            if any(t in {"NewsArticle", "Article", "BlogPosting", "ReportageNewsArticle"} for t in types):
                metadata = item
                break

        raw_text = ""
        try:
            extracted = trafilatura.extract(
                html,
                url=url,
                output_format="txt",
                include_comments=False,
                include_tables=False,
                favor_precision=True,
            )
            raw_text = extracted or ""
        except Exception:
            raw_text = ""

        if len(raw_text.strip()) < 120:
            for selector in BOILERPLATE_SELECTORS:
                for tag in soup.select(selector):
                    tag.decompose()
            candidates = soup.select("article p, main p, .article-content p, .entry-content p, .post-content p, p")
            raw_text = "\n\n".join(p.get_text(" ", strip=True) for p in candidates if len(p.get_text(strip=True)) > 25)

        title = (
            (metadata.get("headline") if isinstance(metadata.get("headline"), str) else None)
            or _first_meta(soup, "og:title", "twitter:title")
            or (soup.title.get_text(" ", strip=True) if soup.title else "")
        )
        author = metadata.get("author")
        if isinstance(author, dict):
            author = author.get("name")
        elif isinstance(author, list) and author and isinstance(author[0], dict):
            author = author[0].get("name")
        if not isinstance(author, str):
            author = _first_meta(soup, "author", "article:author")
        tags = []
        keywords = _first_meta(soup, "keywords", "news_keywords")
        if keywords:
            tags = [tag.strip() for tag in keywords.split(",") if tag.strip()]
        if not raw_text.strip():
            return None
        return Document(
            source=source,
            url=url,
            canonical_url=canonical_url,
            title=title.strip(),
            text=raw_text,
            raw_text=raw_text,
            author=author.strip() if isinstance(author, str) else None,
            published_at=metadata.get("datePublished") or _first_meta(soup, "article:published_time"),
            modified_at=metadata.get("dateModified") or _first_meta(soup, "article:modified_time"),
            category=_first_meta(soup, "article:section"),
            tags=tags,
            subtitle=_first_meta(soup, "description", "og:description"),
        )

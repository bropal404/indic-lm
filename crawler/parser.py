import xml.etree.ElementTree as ET
from typing import Iterable, List, Tuple
from urllib.parse import urljoin

import feedparser
from bs4 import BeautifulSoup


def parse_sitemap(xml_bytes: bytes) -> Tuple[List[str], List[str]]:
    urls: List[str] = []
    sitemaps: List[str] = []
    try:
        soup = BeautifulSoup(xml_bytes, "xml")
        for sm in soup.find_all("sitemap"):
            loc = sm.find("loc")
            if loc and loc.text:
                sitemaps.append(loc.text.strip())
        for u in soup.find_all("url"):
            loc = u.find("loc")
            if loc and loc.text:
                urls.append(loc.text.strip())
        # If standard sitemap/url tags weren't used, check all loc tags
        if not urls and not sitemaps:
            for loc in soup.find_all("loc"):
                if loc.text:
                    txt = loc.text.strip()
                    if "sitemap" in txt.lower() and (txt.endswith(".xml") or ".xml?" in txt):
                        sitemaps.append(txt)
                    else:
                        urls.append(txt)
    except Exception:
        try:
            soup = BeautifulSoup(xml_bytes, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if "sitemap" in href.lower() and href.endswith(".xml"):
                    sitemaps.append(href)
                elif href.startswith("http"):
                    urls.append(href)
        except Exception:
            pass
    return urls, sitemaps


def parse_feed(body: bytes) -> List[str]:
    parsed = feedparser.parse(body)
    return [entry.link for entry in parsed.entries if getattr(entry, "link", None)]


def extract_internal_links(html: str, base_url: str, allowed_domain: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for tag in soup.find_all("a", href=True):
        link = urljoin(base_url, tag["href"])
        if allowed_domain in link:
            links.append(link)
    return links


def detect_page_features(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    return {
        "has_json_ld": bool(soup.find("script", attrs={"type": lambda v: v and "ld+json" in v.lower()})),
        "has_open_graph": bool(soup.find("meta", attrs={"property": lambda v: v and v.startswith("og:")})),
        "has_canonical": bool(soup.find("link", rel=lambda v: v and "canonical" in v)),
        "html_lang": (soup.html.get("lang") if soup.html else None),
        "article_containers": [sel for sel in ["article", "main", ".article-content", ".entry-content", ".post-content"] if soup.select_one(sel)],
    }


def likely_category_links(html: str, base_url: str, allowed_domain: str) -> Iterable[str]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all("a", href=True):
        text = tag.get_text(" ", strip=True).lower()
        href = urljoin(base_url, tag["href"])
        if allowed_domain in href and any(word in text for word in ("समाचार", "खेल", "अर्थ", "विचार", "news", "category")):
            yield href

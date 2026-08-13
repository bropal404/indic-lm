import re
from typing import Iterable, List
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

TRACKING_PREFIXES = ("utm_",)
TRACKING_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src", "spm"}
BLOCKED_RE = re.compile(
    r"(\.(jpg|jpeg|png|gif|webp|svg|ico|css|js|mp3|mp4|mov|zip|rar|pdf|xml|json)$|"
    r"/(login|logout|signin|signup|register|search|tag|tags|trend|topic|author|"
    r"video|videos|photo|photos|photo-gallery|gallery|galleries|audio|podcast|"
    r"cricket|international-cricket-match|international-cricket-team|"
    r"match|matches|scorecard|fixture|fixtures|schedule|"
    r"player|players|team|teams|series|tournament|"
    r"epaper|advertise|contact|about|privacy|terms)(/|$|\?)|"
    r"^(javascript:|mailto:|tel:))",
    re.I,
)


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    path = re.sub(r"/(amp|print)/?$", "", path, flags=re.I) or "/"
    query = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        key_l = key.lower()
        if key_l in TRACKING_KEYS or any(key_l.startswith(prefix) for prefix in TRACKING_PREFIXES):
            continue
        query.append((key, value))
    return urlunparse((scheme, netloc, path, "", urlencode(query, doseq=True), ""))


def is_candidate_url(url: str) -> bool:
    if not url or BLOCKED_RE.search(url):
        return False
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def prioritize_article_urls(urls: Iterable[str]) -> List[str]:
    def score(url: str) -> int:
        path = urlparse(url).path
        points = 0
        if re.search(r"/\d{4}/\d{1,2}/\d{1,2}/", path):
            points += 5
        if re.search(r"[-/]\d{4,}(/|$)", path):
            points += 4
        if len(path.strip("/").split("/")) >= 2:
            points += 2
        if any(word in path.lower() for word in ("news", "story", "article", "samachar")):
            points += 2
        return -points

    return sorted({normalize_url(u) for u in urls if is_candidate_url(u)}, key=score)

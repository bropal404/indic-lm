import urllib.robotparser
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests


@dataclass
class RobotsInfo:
    url: str
    fetched: bool
    status_code: Optional[int]
    sitemaps: List[str]
    crawl_delay: Optional[float]
    disallow_count: int
    error: Optional[str] = None


class RobotsCache:
    def __init__(self, user_agent: str, fallback_allow: bool = False):
        self.user_agent = user_agent
        self.fallback_allow = fallback_allow
        self.parsers: Dict[str, urllib.robotparser.RobotFileParser] = {}
        self.info: Dict[str, RobotsInfo] = {}

    def _base(self, url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def load(self, url: str) -> urllib.robotparser.RobotFileParser:
        base = self._base(url)
        if base in self.parsers:
            return self.parsers[base]
        robots_url = urljoin(base, "/robots.txt")
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)
        sitemaps: List[str] = []
        disallow_count = 0
        try:
            resp = requests.get(robots_url, headers={"User-Agent": self.user_agent}, timeout=15)
            lines = resp.text.splitlines() if resp.ok else []
            parser.parse(lines if lines else (["User-agent: *", "Disallow: /"] if not self.fallback_allow else ["User-agent: *", "Allow: /"]))
            for line in lines:
                low = line.lower()
                if low.startswith("sitemap:"):
                    sitemaps.append(line.split(":", 1)[1].strip())
                if low.startswith("disallow:") and line.split(":", 1)[1].strip():
                    disallow_count += 1
            delay = parser.crawl_delay(self.user_agent) or parser.crawl_delay("*")
            self.info[base] = RobotsInfo(robots_url, bool(lines), resp.status_code, sitemaps, float(delay) if delay else None, disallow_count)
        except Exception as exc:
            parser.parse(["User-agent: *", "Disallow: /"] if not self.fallback_allow else ["User-agent: *", "Allow: /"])
            self.info[base] = RobotsInfo(robots_url, False, None, [], None, 0, str(exc))
        self.parsers[base] = parser
        return parser

    def can_fetch(self, url: str) -> bool:
        return self.load(url).can_fetch(self.user_agent, url)

    def crawl_delay(self, url: str) -> Optional[float]:
        self.load(url)
        return self.info[self._base(url)].crawl_delay

    def get_info(self, url: str) -> RobotsInfo:
        self.load(url)
        return self.info[self._base(url)]

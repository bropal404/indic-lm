import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlparse

import aiohttp

from .normalize import repair_mojibake


@dataclass
class FetchResult:
    url: str
    status: int
    content_type: str
    body: bytes
    final_url: str

    @property
    def text(self) -> str:
        return repair_mojibake(self.body.decode("utf-8", errors="replace"))


class Fetcher:
    def __init__(self, user_agent: str, timeout: int = 30):
        self.user_agent = user_agent
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_request: Dict[str, float] = {}

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            limit=0,              # no global cap — rate-limiting is per-domain in fetch()
            limit_per_host=32,    # max 32 simultaneous connections to any single host
            ttl_dns_cache=300,    # cache DNS for 5 min to reduce resolver load
            enable_cleanup_closed=True,
        )
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout,
            headers={"User-Agent": self.user_agent},
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()

    async def fetch(self, url: str, requests_per_second: float, max_retries: int = 3) -> Optional[FetchResult]:
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=self.timeout, headers={"User-Agent": self.user_agent})
        domain = urlparse(url).netloc
        delay = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        backoff = 1.0
        for attempt in range(max_retries + 1):
            elapsed = time.monotonic() - self.last_request.get(domain, 0.0)
            if delay and elapsed < delay:
                await asyncio.sleep(delay - elapsed)
            self.last_request[domain] = time.monotonic()
            try:
                async with self.session.get(url, allow_redirects=True) as resp:
                    body = await resp.read()
                    if resp.status == 200:
                        return FetchResult(url, resp.status, resp.headers.get("Content-Type", ""), body, str(resp.url))
                    if resp.status not in {408, 425, 429, 500, 502, 503, 504}:
                        return FetchResult(url, resp.status, resp.headers.get("Content-Type", ""), b"", str(resp.url))
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            if attempt < max_retries:
                await asyncio.sleep(backoff)
                backoff *= 2
        return None

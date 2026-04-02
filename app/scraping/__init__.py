"""Advanced web scraping infrastructure with anti-detection and compliance.

Imports are **lazy** — Playwright, Selenium, and other browser dependencies are
only pulled in when a specific symbol is first accessed.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from app.scraping.manager import ScrapingManager
    from app.scraping.base import BaseScraper, ScrapedContent, ScraperConfig

__all__ = [
    "ScrapingManager",
    "BaseScraper",
    "ScrapedContent",
    "ScraperConfig",
]

_MODULE_MAP: dict[str, str] = {
    "ScrapingManager": "app.scraping.manager",
    "BaseScraper":     "app.scraping.base",
    "ScrapedContent":  "app.scraping.base",
    "ScraperConfig":   "app.scraping.base",
}


def __getattr__(name: str):  # noqa: ANN001, ANN201
    if name in _MODULE_MAP:
        module = importlib.import_module(_MODULE_MAP[name])
        obj = getattr(module, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module 'app.scraping' has no attribute {name!r}")

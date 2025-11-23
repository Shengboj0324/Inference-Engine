"""Browser fingerprint randomization for anti-detection."""

import random
from typing import Any, Dict, List


class BrowserFingerprint:
    """Generate randomized browser fingerprints."""

    USER_AGENTS = [
        # Chrome on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        # Chrome on macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        # Firefox on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
        # Firefox on macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
        # Safari on macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
        # Edge on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    ]

    SCREEN_RESOLUTIONS = [
        {"width": 1920, "height": 1080},
        {"width": 1366, "height": 768},
        {"width": 1536, "height": 864},
        {"width": 1440, "height": 900},
        {"width": 2560, "height": 1440},
        {"width": 1280, "height": 720},
        {"width": 1600, "height": 900},
    ]

    TIMEZONES = [
        "America/New_York",
        "America/Chicago",
        "America/Los_Angeles",
        "America/Denver",
        "Europe/London",
        "Europe/Paris",
        "Europe/Berlin",
        "Asia/Tokyo",
        "Asia/Shanghai",
        "Australia/Sydney",
    ]

    LOCALES = [
        "en-US",
        "en-GB",
        "en-CA",
        "en-AU",
        "de-DE",
        "fr-FR",
        "es-ES",
        "ja-JP",
        "zh-CN",
    ]

    async def generate_context_options(
        self, randomize: bool = True
    ) -> Dict[str, Any]:
        """Generate browser context options with fingerprint.

        Args:
            randomize: Whether to randomize fingerprint

        Returns:
            Context options dictionary
        """
        if not randomize:
            return {
                "user_agent": self.USER_AGENTS[0],
                "viewport": self.SCREEN_RESOLUTIONS[0],
                "locale": "en-US",
                "timezone_id": "America/New_York",
            }

        # Random user agent
        user_agent = random.choice(self.USER_AGENTS)

        # Random screen resolution
        resolution = random.choice(self.SCREEN_RESOLUTIONS)

        # Random timezone
        timezone = random.choice(self.TIMEZONES)

        # Random locale
        locale = random.choice(self.LOCALES)

        # Random color depth
        color_depth = random.choice([24, 32])

        # Random device scale factor
        device_scale_factor = random.choice([1, 1.5, 2])

        return {
            "user_agent": user_agent,
            "viewport": resolution,
            "locale": locale,
            "timezone_id": timezone,
            "device_scale_factor": device_scale_factor,
            "has_touch": random.choice([True, False]),
            "is_mobile": False,
            "java_script_enabled": True,
            "permissions": ["geolocation", "notifications"],
            "extra_http_headers": {
                "Accept-Language": f"{locale},en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            },
        }

    def get_random_user_agent(self) -> str:
        """Get a random user agent string."""
        return random.choice(self.USER_AGENTS)

    def get_random_headers(self) -> Dict[str, str]:
        """Get random HTTP headers."""
        locale = random.choice(self.LOCALES)

        return {
            "User-Agent": self.get_random_user_agent(),
            "Accept-Language": f"{locale},en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0",
        }


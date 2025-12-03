"""Industrial-grade human behavior simulation for anti-bot evasion.

Implements Bézier curve mouse movement, realistic typing patterns, and human-like delays
to bypass sophisticated anti-bot detection systems (TikTok, Instagram, etc.).
"""

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import List, Tuple

from playwright.async_api import Page

logger = logging.getLogger(__name__)


@dataclass
class Point:
    """2D point for mouse movement."""

    x: float
    y: float


class BezierCurve:
    """Bézier curve generator for smooth, human-like mouse paths."""

    @staticmethod
    def cubic_bezier(
        start: Point,
        end: Point,
        control1: Point,
        control2: Point,
        num_points: int = 50,
    ) -> List[Point]:
        """Generate points along a cubic Bézier curve.

        Args:
            start: Starting point
            end: Ending point
            control1: First control point
            control2: Second control point
            num_points: Number of points to generate

        Returns:
            List of points along the curve
        """
        points = []

        for i in range(num_points + 1):
            t = i / num_points

            # Cubic Bézier formula: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
            x = (
                (1 - t) ** 3 * start.x
                + 3 * (1 - t) ** 2 * t * control1.x
                + 3 * (1 - t) * t**2 * control2.x
                + t**3 * end.x
            )

            y = (
                (1 - t) ** 3 * start.y
                + 3 * (1 - t) ** 2 * t * control1.y
                + 3 * (1 - t) * t**2 * control2.y
                + t**3 * end.y
            )

            points.append(Point(x, y))

        return points

    @staticmethod
    def generate_control_points(
        start: Point,
        end: Point,
        curvature: float = 0.3,
    ) -> Tuple[Point, Point]:
        """Generate random control points for natural-looking curves.

        Args:
            start: Starting point
            end: Ending point
            curvature: Curve intensity (0=straight, 1=very curved)

        Returns:
            Tuple of (control1, control2)
        """
        # Calculate midpoint
        mid_x = (start.x + end.x) / 2
        mid_y = (start.y + end.y) / 2

        # Calculate perpendicular offset
        dx = end.x - start.x
        dy = end.y - start.y
        distance = (dx**2 + dy**2) ** 0.5

        # Random offset perpendicular to the line
        offset = distance * curvature * random.uniform(-1, 1)

        # Perpendicular vector
        perp_x = -dy / distance if distance > 0 else 0
        perp_y = dx / distance if distance > 0 else 0

        # Control points offset from midpoint
        control1 = Point(
            mid_x + perp_x * offset * random.uniform(0.3, 0.7),
            mid_y + perp_y * offset * random.uniform(0.3, 0.7),
        )

        control2 = Point(
            mid_x + perp_x * offset * random.uniform(0.3, 0.7),
            mid_y + perp_y * offset * random.uniform(0.3, 0.7),
        )

        return control1, control2


class HumanSimulator:
    """Simulate human-like interactions for anti-bot evasion.

    Features:
    - Bézier curve mouse movement
    - Realistic typing with errors and corrections
    - Human-like delays and pauses
    - Random micro-movements
    - Scroll behavior simulation
    """

    def __init__(
        self,
        typing_speed_wpm: int = 60,
        error_rate: float = 0.05,
        mouse_speed_pixels_per_second: int = 800,
    ):
        """Initialize human simulator.

        Args:
            typing_speed_wpm: Typing speed in words per minute
            error_rate: Probability of typing errors (0-1)
            mouse_speed_pixels_per_second: Mouse movement speed
        """
        self.typing_speed_wpm = typing_speed_wpm
        self.error_rate = error_rate
        self.mouse_speed = mouse_speed_pixels_per_second

        # Calculate typing delay (characters per second)
        # Average word length is 5 characters
        self.chars_per_second = (typing_speed_wpm * 5) / 60

    async def move_mouse_to(
        self,
        page: Page,
        target_x: float,
        target_y: float,
        curvature: float = 0.3,
        num_steps: int = 50,
    ) -> None:
        """Move mouse to target using Bézier curve.

        Args:
            page: Playwright page
            target_x: Target X coordinate
            target_y: Target Y coordinate
            curvature: Curve intensity (0=straight, 1=very curved)
            num_steps: Number of movement steps
        """
        # Get current mouse position (approximate from viewport center)
        viewport = page.viewport_size
        if not viewport:
            viewport = {"width": 1920, "height": 1080}

        current_x = viewport["width"] / 2
        current_y = viewport["height"] / 2

        start = Point(current_x, current_y)
        end = Point(target_x, target_y)

        # Generate control points
        control1, control2 = BezierCurve.generate_control_points(
            start, end, curvature
        )

        # Generate curve points
        points = BezierCurve.cubic_bezier(start, end, control1, control2, num_steps)

        # Calculate total distance for timing
        total_distance = ((end.x - start.x) ** 2 + (end.y - start.y) ** 2) ** 0.5
        total_time = total_distance / self.mouse_speed

        # Move along curve
        for i, point in enumerate(points):
            await page.mouse.move(point.x, point.y)

            # Variable delay (faster in middle, slower at start/end)
            progress = i / len(points)
            speed_multiplier = 1.0 + 0.5 * (1 - abs(2 * progress - 1))  # Ease in/out
            delay = (total_time / num_steps) * speed_multiplier

            # Add small random jitter
            delay += random.uniform(-0.001, 0.001)

            await asyncio.sleep(max(delay, 0.001))

        logger.debug(f"Moved mouse to ({target_x}, {target_y}) in {total_time:.2f}s")

    async def type_like_human(
        self,
        page: Page,
        selector: str,
        text: str,
        make_errors: bool = True,
    ) -> None:
        """Type text with human-like patterns.

        Args:
            page: Playwright page
            selector: Element selector to type into
            text: Text to type
            make_errors: Whether to simulate typing errors
        """
        # Click on element first
        await page.click(selector)
        await asyncio.sleep(random.uniform(0.1, 0.3))

        for i, char in enumerate(text):
            # Simulate typing error
            if make_errors and random.random() < self.error_rate:
                # Type wrong character
                wrong_char = random.choice("abcdefghijklmnopqrstuvwxyz")
                await page.keyboard.type(wrong_char)
                await asyncio.sleep(random.uniform(0.1, 0.3))

                # Pause (realize mistake)
                await asyncio.sleep(random.uniform(0.2, 0.5))

                # Delete wrong character
                await page.keyboard.press("Backspace")
                await asyncio.sleep(random.uniform(0.05, 0.15))

            # Type correct character
            await page.keyboard.type(char)

            # Variable delay based on character type
            if char == " ":
                # Longer pause after space (word boundary)
                delay = random.uniform(0.15, 0.35)
            elif char in ".,!?;:":
                # Pause after punctuation
                delay = random.uniform(0.2, 0.4)
            else:
                # Normal typing delay
                base_delay = 1.0 / self.chars_per_second
                delay = random.gauss(base_delay, base_delay * 0.3)
                delay = max(0.05, delay)  # Minimum delay

            await asyncio.sleep(delay)

        logger.debug(f"Typed '{text}' with human-like pattern")

    async def scroll_like_human(
        self,
        page: Page,
        scroll_amount: int = 500,
        num_scrolls: int = 3,
    ) -> None:
        """Scroll page with human-like behavior.

        Args:
            page: Playwright page
            scroll_amount: Pixels to scroll each time
            num_scrolls: Number of scroll actions
        """
        for i in range(num_scrolls):
            # Variable scroll amount
            amount = scroll_amount + random.randint(-100, 100)

            # Scroll
            await page.mouse.wheel(0, amount)

            # Pause to "read" content
            pause = random.uniform(0.5, 2.0)
            await asyncio.sleep(pause)

            # Occasional scroll back up (realistic behavior)
            if random.random() < 0.2:
                await page.mouse.wheel(0, -random.randint(50, 150))
                await asyncio.sleep(random.uniform(0.3, 0.8))

        logger.debug(f"Scrolled {num_scrolls} times with human-like pattern")

    async def random_micro_movement(self, page: Page) -> None:
        """Perform small random mouse movements (human fidgeting).

        Args:
            page: Playwright page
        """
        # Small random movement
        offset_x = random.randint(-20, 20)
        offset_y = random.randint(-20, 20)

        viewport = page.viewport_size
        if not viewport:
            return

        current_x = viewport["width"] / 2
        current_y = viewport["height"] / 2

        await self.move_mouse_to(
            page,
            current_x + offset_x,
            current_y + offset_y,
            curvature=0.1,
            num_steps=10,
        )

    async def human_delay(
        self,
        min_seconds: float = 0.5,
        max_seconds: float = 2.0,
    ) -> None:
        """Wait with human-like delay.

        Args:
            min_seconds: Minimum delay
            max_seconds: Maximum delay
        """
        delay = random.uniform(min_seconds, max_seconds)
        await asyncio.sleep(delay)

    async def simulate_reading(
        self,
        page: Page,
        content_length: int = 1000,
    ) -> None:
        """Simulate reading content (pause based on length).

        Args:
            page: Playwright page
            content_length: Approximate content length in characters
        """
        # Average reading speed: 200-250 words per minute
        # Average word length: 5 characters
        words = content_length / 5
        reading_time = (words / 225) * 60  # seconds

        # Add randomness
        reading_time *= random.uniform(0.7, 1.3)

        # Minimum 2 seconds, maximum 30 seconds
        reading_time = max(2.0, min(reading_time, 30.0))

        logger.debug(f"Simulating reading for {reading_time:.1f}s")
        await asyncio.sleep(reading_time)


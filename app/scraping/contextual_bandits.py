"""Industrial-grade contextual bandits for intelligent proxy rotation.

Implements UCB1 (Upper Confidence Bound) algorithm to learn optimal proxy/IP routing
per platform, reducing blocks and maximizing success rates.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProxyArm:
    """Represents a proxy as a bandit arm."""

    proxy_id: str
    host: str
    port: int
    country: Optional[str] = None

    # UCB1 statistics
    total_pulls: int = 0
    total_successes: int = 0
    total_failures: int = 0
    success_rate: float = 0.0
    ucb_score: float = float("inf")  # Start with infinite to ensure initial exploration

    # Platform-specific statistics
    platform_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Performance metrics
    avg_response_time: float = 0.0
    last_used: Optional[datetime] = None
    consecutive_failures: int = 0
    is_blocked: bool = False


@dataclass
class BanditContext:
    """Context for contextual bandit decision."""

    platform: str
    content_type: str
    time_of_day: int  # 0-23
    is_trending: bool = False
    requires_auth: bool = False


class UCB1ProxySelector:
    """UCB1 (Upper Confidence Bound) algorithm for proxy selection.

    Features:
    - Platform-specific learning (Reddit vs TikTok)
    - Exploration-exploitation balance
    - Automatic blocking detection
    - Performance-based selection
    - Context-aware routing
    """

    def __init__(
        self,
        exploration_factor: float = 2.0,
        min_pulls_before_exploitation: int = 10,
        block_threshold: int = 3,
        block_cooldown_seconds: int = 300,
    ):
        """Initialize UCB1 proxy selector.

        Args:
            exploration_factor: UCB1 exploration parameter (higher = more exploration)
            min_pulls_before_exploitation: Minimum pulls before using UCB scores
            block_threshold: Consecutive failures before marking as blocked
            block_cooldown_seconds: Seconds to wait before retrying blocked proxy
        """
        self.exploration_factor = exploration_factor
        self.min_pulls_before_exploitation = min_pulls_before_exploitation
        self.block_threshold = block_threshold
        self.block_cooldown_seconds = block_cooldown_seconds

        # Proxy arms
        self.arms: Dict[str, ProxyArm] = {}

        # Global statistics
        self.total_pulls = 0
        self.total_successes = 0
        self.total_failures = 0

    def add_proxy(
        self,
        proxy_id: str,
        host: str,
        port: int,
        country: Optional[str] = None,
    ) -> None:
        """Add proxy to the bandit.

        Args:
            proxy_id: Unique proxy identifier
            host: Proxy host
            port: Proxy port
            country: Proxy country (optional)
        """
        if proxy_id in self.arms:
            logger.warning(f"Proxy {proxy_id} already exists")
            return

        self.arms[proxy_id] = ProxyArm(
            proxy_id=proxy_id,
            host=host,
            port=port,
            country=country,
        )

        logger.info(f"Added proxy: {proxy_id} ({host}:{port})")

    def select_proxy(self, context: Optional[BanditContext] = None) -> Optional[ProxyArm]:
        """Select best proxy using UCB1 algorithm.

        Args:
            context: Context for contextual selection

        Returns:
            Selected proxy arm, or None if no proxies available
        """
        if not self.arms:
            logger.warning("No proxies available")
            return None

        # Filter out blocked proxies
        available_arms = [
            arm for arm in self.arms.values()
            if not self._is_currently_blocked(arm)
        ]

        if not available_arms:
            logger.warning("All proxies are blocked")
            return None

        # Update UCB scores
        self._update_ucb_scores(available_arms, context)

        # Select arm with highest UCB score
        selected = max(available_arms, key=lambda arm: arm.ucb_score)

        logger.debug(
            f"Selected proxy: {selected.proxy_id} "
            f"(UCB={selected.ucb_score:.4f}, success_rate={selected.success_rate:.2%})"
        )

        return selected

    def update_reward(
        self,
        proxy_id: str,
        success: bool,
        response_time: Optional[float] = None,
        context: Optional[BanditContext] = None,
    ) -> None:
        """Update proxy statistics after use.

        Args:
            proxy_id: Proxy that was used
            success: Whether the request succeeded
            response_time: Response time in seconds
            context: Context of the request
        """
        if proxy_id not in self.arms:
            logger.warning(f"Unknown proxy: {proxy_id}")
            return

        arm = self.arms[proxy_id]

        # Update global statistics
        self.total_pulls += 1
        arm.total_pulls += 1
        arm.last_used = datetime.utcnow()

        if success:
            self.total_successes += 1
            arm.total_successes += 1
            arm.consecutive_failures = 0
            arm.is_blocked = False
        else:
            self.total_failures += 1
            arm.total_failures += 1
            arm.consecutive_failures += 1

            # Check if proxy should be marked as blocked
            if arm.consecutive_failures >= self.block_threshold:
                arm.is_blocked = True
                logger.warning(
                    f"Proxy {proxy_id} marked as blocked "
                    f"({arm.consecutive_failures} consecutive failures)"
                )

        # Update success rate
        if arm.total_pulls > 0:
            arm.success_rate = arm.total_successes / arm.total_pulls

        # Update response time (exponential moving average)
        if response_time is not None:
            if arm.avg_response_time == 0:
                arm.avg_response_time = response_time
            else:
                alpha = 0.3  # Smoothing factor
                arm.avg_response_time = (
                    alpha * response_time + (1 - alpha) * arm.avg_response_time
                )

        # Update platform-specific statistics
        if context and context.platform:
            if context.platform not in arm.platform_stats:
                arm.platform_stats[context.platform] = {
                    "pulls": 0,
                    "successes": 0,
                    "failures": 0,
                }

            stats = arm.platform_stats[context.platform]
            stats["pulls"] += 1
            if success:
                stats["successes"] += 1
            else:
                stats["failures"] += 1

        logger.debug(
            f"Updated proxy {proxy_id}: "
            f"success_rate={arm.success_rate:.2%}, "
            f"pulls={arm.total_pulls}"
        )

    def _update_ucb_scores(
        self,
        arms: List[ProxyArm],
        context: Optional[BanditContext] = None,
    ) -> None:
        """Update UCB scores for all arms.

        UCB1 formula: UCB = mean_reward + sqrt(c * ln(total_pulls) / arm_pulls)
        """
        for arm in arms:
            # Use platform-specific statistics if context provided
            if context and context.platform in arm.platform_stats:
                stats = arm.platform_stats[context.platform]
                arm_pulls = stats["pulls"]
                arm_successes = stats["successes"]
                mean_reward = arm_successes / arm_pulls if arm_pulls > 0 else 0
            else:
                arm_pulls = arm.total_pulls
                mean_reward = arm.success_rate

            # Ensure minimum exploration
            if arm_pulls < self.min_pulls_before_exploitation:
                arm.ucb_score = float("inf")
                continue

            # Calculate UCB score
            if arm_pulls > 0 and self.total_pulls > 0:
                exploration_bonus = math.sqrt(
                    (self.exploration_factor * math.log(self.total_pulls)) / arm_pulls
                )
                arm.ucb_score = mean_reward + exploration_bonus
            else:
                arm.ucb_score = float("inf")

    def _is_currently_blocked(self, arm: ProxyArm) -> bool:
        """Check if proxy is currently blocked.

        Args:
            arm: Proxy arm to check

        Returns:
            True if blocked, False otherwise
        """
        if not arm.is_blocked:
            return False

        # Check if cooldown period has passed
        if arm.last_used:
            elapsed = (datetime.utcnow() - arm.last_used).total_seconds()
            if elapsed >= self.block_cooldown_seconds:
                # Reset block status after cooldown
                arm.is_blocked = False
                arm.consecutive_failures = 0
                logger.info(f"Proxy {arm.proxy_id} unblocked after cooldown")
                return False

        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get bandit statistics.

        Returns:
            Dictionary with statistics
        """
        active_proxies = sum(1 for arm in self.arms.values() if not arm.is_blocked)
        blocked_proxies = sum(1 for arm in self.arms.values() if arm.is_blocked)

        # Get best performing proxy
        best_proxy = None
        best_rate = 0.0
        for arm in self.arms.values():
            if arm.total_pulls >= self.min_pulls_before_exploitation:
                if arm.success_rate > best_rate:
                    best_rate = arm.success_rate
                    best_proxy = arm.proxy_id

        return {
            "total_proxies": len(self.arms),
            "active_proxies": active_proxies,
            "blocked_proxies": blocked_proxies,
            "total_pulls": self.total_pulls,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "overall_success_rate": (
                self.total_successes / self.total_pulls if self.total_pulls > 0 else 0
            ),
            "best_proxy": best_proxy,
            "best_success_rate": best_rate,
        }

    def get_proxy_statistics(self, proxy_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific proxy.

        Args:
            proxy_id: Proxy identifier

        Returns:
            Dictionary with proxy statistics, or None if not found
        """
        if proxy_id not in self.arms:
            return None

        arm = self.arms[proxy_id]

        return {
            "proxy_id": arm.proxy_id,
            "host": arm.host,
            "port": arm.port,
            "country": arm.country,
            "total_pulls": arm.total_pulls,
            "total_successes": arm.total_successes,
            "total_failures": arm.total_failures,
            "success_rate": arm.success_rate,
            "ucb_score": arm.ucb_score,
            "avg_response_time": arm.avg_response_time,
            "consecutive_failures": arm.consecutive_failures,
            "is_blocked": arm.is_blocked,
            "last_used": arm.last_used.isoformat() if arm.last_used else None,
            "platform_stats": arm.platform_stats,
        }

    def reset_proxy(self, proxy_id: str) -> bool:
        """Reset statistics for a proxy.

        Args:
            proxy_id: Proxy to reset

        Returns:
            True if reset, False if proxy not found
        """
        if proxy_id not in self.arms:
            return False

        arm = self.arms[proxy_id]
        arm.total_pulls = 0
        arm.total_successes = 0
        arm.total_failures = 0
        arm.success_rate = 0.0
        arm.ucb_score = float("inf")
        arm.consecutive_failures = 0
        arm.is_blocked = False
        arm.platform_stats.clear()

        logger.info(f"Reset proxy statistics: {proxy_id}")
        return True


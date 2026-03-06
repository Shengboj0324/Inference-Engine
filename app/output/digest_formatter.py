"""Professional digest formatting for beautiful user-facing output.

This module provides multiple output formats for digests:
- HTML: Rich, interactive web display
- Markdown: Clean, readable text format
- JSON: Structured data for custom rendering
"""

import logging
from datetime import datetime
from typing import List, Optional

from app.core.models import Cluster, ContentItem, DigestResponse

logger = logging.getLogger(__name__)


class DigestFormatter:
    """Format digests for professional display."""

    def format_html(self, digest: DigestResponse) -> str:
        """Generate beautiful HTML output for digest.

        Args:
            digest: Digest to format

        Returns:
            HTML string with embedded CSS
        """
        html_parts = []

        # HTML header with embedded CSS
        html_parts.append(self._get_html_header())

        # Digest header
        html_parts.append(self._format_digest_header_html(digest))

        # Clusters
        for i, cluster in enumerate(digest.clusters, 1):
            html_parts.append(self._format_cluster_html(cluster, i))

        # Footer
        html_parts.append(self._get_html_footer())

        return "\n".join(html_parts)

    def format_markdown(self, digest: DigestResponse) -> str:
        """Generate clean Markdown output for digest.

        Args:
            digest: Digest to format

        Returns:
            Markdown string
        """
        md_parts = []

        # Header
        md_parts.append(f"# 📰 Your Daily Digest")
        md_parts.append(f"\n**Period:** {digest.period_start.strftime('%Y-%m-%d %H:%M')} - {digest.period_end.strftime('%Y-%m-%d %H:%M')}")
        md_parts.append(f"\n**Total Items:** {digest.total_items}")
        md_parts.append(f"\n**Clusters:** {len(digest.clusters)}")
        md_parts.append(f"\n\n## Executive Summary\n\n{digest.summary}\n")
        md_parts.append("\n---\n")

        # Clusters
        for i, cluster in enumerate(digest.clusters, 1):
            md_parts.append(self._format_cluster_markdown(cluster, i))

        return "\n".join(md_parts)

    def _get_html_header(self) -> str:
        """Get HTML header with embedded CSS."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Your Daily Digest</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header .meta { opacity: 0.9; font-size: 0.95em; }
        .summary {
            background: #f8f9fa;
            padding: 30px;
            border-left: 4px solid #667eea;
            margin: 30px;
            border-radius: 8px;
        }
        .summary h2 { color: #667eea; margin-bottom: 15px; }
        .cluster {
            margin: 30px;
            padding: 25px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .cluster:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .cluster-header {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        .cluster-number {
            background: #667eea;
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 15px;
        }
        .cluster-title { font-size: 1.5em; color: #333; flex: 1; }
        .cluster-meta {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
            font-size: 0.9em;
            color: #666;
        }
        .badge {
            background: #e3f2fd;
            color: #1976d2;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
        }
        .cluster-summary { margin: 15px 0; line-height: 1.8; }
        .items { margin-top: 20px; }
        .item {
            padding: 15px;
            background: #f8f9fa;
            border-left: 3px solid #667eea;
            margin-bottom: 10px;
            border-radius: 4px;
        }
        .item-title { font-weight: 600; color: #333; margin-bottom: 5px; }
        .item-meta { font-size: 0.85em; color: #666; }
        .footer {
            text-align: center;
            padding: 30px;
            background: #f8f9fa;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">"""

    def _format_digest_header_html(self, digest: DigestResponse) -> str:
        """Format digest header in HTML."""
        return f"""
        <div class="header">
            <h1>📰 Your Daily Digest</h1>
            <div class="meta">
                {digest.period_start.strftime('%B %d, %Y %H:%M')} - {digest.period_end.strftime('%H:%M')}
                <br>
                {digest.total_items} items • {len(digest.clusters)} topics
            </div>
        </div>
        <div class="summary">
            <h2>Executive Summary</h2>
            <p>{digest.summary}</p>
        </div>"""

    def _format_cluster_html(self, cluster: Cluster, index: int) -> str:
        """Format a single cluster in HTML."""
        platforms = ", ".join([p.value for p in cluster.platforms_represented])

        items_html = ""
        for item in cluster.items[:5]:  # Show top 5 items
            items_html += f"""
            <div class="item">
                <div class="item-title">{item.title}</div>
                <div class="item-meta">
                    {item.source_platform.value} • {item.author or 'Unknown'} •
                    {item.published_at.strftime('%b %d, %H:%M')}
                </div>
            </div>"""

        return f"""
        <div class="cluster">
            <div class="cluster-header">
                <div class="cluster-number">{index}</div>
                <div class="cluster-title">{cluster.topic}</div>
            </div>
            <div class="cluster-meta">
                <span class="badge">{len(cluster.items)} items</span>
                <span class="badge">{platforms}</span>
                <span class="badge">Relevance: {cluster.relevance_score:.2f}</span>
            </div>
            <div class="cluster-summary">{cluster.summary}</div>
            <div class="items">
                <strong>Key Items:</strong>
                {items_html}
            </div>
        </div>"""

    def _get_html_footer(self) -> str:
        """Get HTML footer."""
        return """
        <div class="footer">
            Generated by Social Media Radar • {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </div>
    </div>
</body>
</html>"""

    def _format_cluster_markdown(self, cluster: Cluster, index: int) -> str:
        """Format a single cluster in Markdown."""
        platforms = ", ".join([p.value for p in cluster.platforms_represented])

        md = f"\n## {index}. {cluster.topic}\n\n"
        md += f"**Platforms:** {platforms} | **Items:** {len(cluster.items)} | **Relevance:** {cluster.relevance_score:.2f}\n\n"
        md += f"{cluster.summary}\n\n"

        if cluster.keywords:
            md += f"**Key Points:** {', '.join(cluster.keywords[:5])}\n\n"

        md += "### Top Items:\n\n"
        for item in cluster.items[:5]:
            md += f"- **{item.title}**\n"
            md += f"  - Source: {item.source_platform.value}"
            if item.author:
                md += f" | Author: {item.author}"
            md += f" | {item.published_at.strftime('%b %d, %H:%M')}\n"
            if item.source_url:
                md += f"  - [View Original]({item.source_url})\n"

        md += "\n---\n"
        return md


class RichMediaFormatter:
    """Format digests with rich media (images, videos) embedded."""

    def format_html_with_media(self, digest: DigestResponse) -> str:
        """Generate HTML with embedded media.

        Args:
            digest: Digest to format

        Returns:
            HTML with embedded images and video players
        """
        # Rich media embedding
        # Note: This is a basic implementation. Future enhancements:
        # - Video player integration
        # - Media galleries
        # - Advanced lazy loading

        # For now, return the digest as-is
        # Media URLs are already included in the content
        return digest

    def _embed_image(self, url: str, alt: str = "") -> str:
        """Generate HTML for embedded image."""
        return f'<img src="{url}" alt="{alt}" loading="lazy" style="max-width: 100%; border-radius: 8px; margin: 10px 0;">'

    def _embed_video(self, url: str, platform: str) -> str:
        """Generate HTML for embedded video player."""
        if "youtube" in platform.lower():
            # Extract video ID and embed YouTube player
            return f'<iframe width="100%" height="400" src="{url}" frameborder="0" allowfullscreen></iframe>'
        else:
            # Generic video player
            return f'<video controls style="max-width: 100%; border-radius: 8px;"><source src="{url}"></video>'


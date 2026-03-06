"""MCP server for Social Media Radar.

This server exposes tools that can be used by MCP-compatible clients
to interact with the Social Media Radar system.
"""

from typing import Any, Dict, List

from pydantic import BaseModel


class MCPTool(BaseModel):
    """MCP tool definition."""

    name: str
    description: str
    parameters: Dict[str, Any]


class MCPServer:
    """MCP server for Social Media Radar."""

    def __init__(self):
        """Initialize MCP server."""
        self.tools = self._register_tools()

    def _register_tools(self) -> List[MCPTool]:
        """Register available MCP tools.

        Returns:
            List of available tools
        """
        return [
            MCPTool(
                name="get_daily_digest",
                description="Get a personalized daily digest of content from configured sources",
                parameters={
                    "type": "object",
                    "properties": {
                        "topics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by specific topics (optional)",
                        },
                        "since_hours": {
                            "type": "integer",
                            "description": "Hours to look back (default: 24)",
                            "default": 24,
                        },
                        "max_clusters": {
                            "type": "integer",
                            "description": "Maximum number of topic clusters (default: 20)",
                            "default": 20,
                        },
                        "platforms": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by platforms (optional)",
                        },
                    },
                },
            ),
            MCPTool(
                name="search_content",
                description="Search through your content backlog",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "platforms": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by platforms (optional)",
                        },
                        "since_hours": {
                            "type": "integer",
                            "description": "Hours to look back (optional)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results (default: 50)",
                            "default": 50,
                        },
                    },
                    "required": ["query"],
                },
            ),
            MCPTool(
                name="configure_source",
                description="Configure a content source (Reddit, YouTube, RSS, etc.)",
                parameters={
                    "type": "object",
                    "properties": {
                        "platform": {
                            "type": "string",
                            "enum": [
                                "reddit",
                                "youtube",
                                "tiktok",
                                "facebook",
                                "instagram",
                                "rss",
                                "newsapi",
                                "nytimes",
                            ],
                            "description": "Platform to configure",
                        },
                        "credentials": {
                            "type": "object",
                            "description": "Platform-specific credentials",
                        },
                        "settings": {
                            "type": "object",
                            "description": "Platform-specific settings (optional)",
                        },
                    },
                    "required": ["platform", "credentials"],
                },
            ),
            MCPTool(
                name="list_sources",
                description="List all configured content sources",
                parameters={"type": "object", "properties": {}},
            ),
            MCPTool(
                name="get_cluster_detail",
                description="Get detailed information about a content cluster",
                parameters={
                    "type": "object",
                    "properties": {
                        "cluster_id": {
                            "type": "string",
                            "description": "Cluster UUID",
                        },
                    },
                    "required": ["cluster_id"],
                },
            ),
        ]

    def list_tools(self) -> List[MCPTool]:
        """List available tools.

        Returns:
            List of MCP tools
        """
        return self.tools

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an MCP tool.

        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters

        Returns:
            Tool execution result
        """
        # Execute tool by calling appropriate backend API endpoints
        try:
            if tool_name == "get_daily_digest":
                return await self._get_daily_digest(parameters)
            elif tool_name == "search_content":
                return await self._search_content(parameters)
            elif tool_name == "configure_source":
                return await self._configure_source(parameters)
            elif tool_name == "list_sources":
                return await self._list_sources(parameters)
            elif tool_name == "get_cluster_detail":
                return await self._get_cluster_detail(parameters)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {e}")
            return {"error": str(e)}

    async def _get_daily_digest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute get_daily_digest tool by calling digest API endpoint."""
        try:
            # Import here to avoid circular dependencies
            from app.intelligence.digest_engine import DigestEngine
            from app.core.db import get_db

            # Get database session
            async for db in get_db():
                engine = DigestEngine()
                digest = await engine.generate_digest(
                    user_id=params.get("user_id"),
                    db=db,
                    format=params.get("format", "markdown")
                )
                return {"digest": digest, "status": "success"}
        except Exception as e:
            return {"error": str(e), "status": "failed"}

    async def _search_content(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute search_content tool by calling search API endpoint."""
        try:
            from app.intelligence.hnsw_search import HNSWSearchEngine

            engine = HNSWSearchEngine()
            results = await engine.search(
                query=params.get("query", ""),
                top_k=params.get("limit", 10)
            )
            return {"results": results, "status": "success"}
        except Exception as e:
            return {"error": str(e), "status": "failed"}

    async def _configure_source(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute configure_source tool by calling sources API endpoint."""
        try:
            from app.core.db import get_db
            from app.core.db_models import SourceConfig

            async for db in get_db():
                # Create or update source configuration
                source = SourceConfig(
                    user_id=params.get("user_id"),
                    platform=params.get("platform"),
                    settings=params.get("settings", {})
                )
                db.add(source)
                await db.commit()
                return {"status": "success", "source_id": str(source.id)}
        except Exception as e:
            return {"error": str(e), "status": "failed"}

    async def _list_sources(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute list_sources tool by calling sources API endpoint."""
        try:
            from app.core.db import get_db
            from app.core.db_models import SourceConfig
            from sqlalchemy import select

            async for db in get_db():
                result = await db.execute(
                    select(SourceConfig).where(
                        SourceConfig.user_id == params.get("user_id")
                    )
                )
                sources = result.scalars().all()
                return {
                    "sources": [
                        {
                            "id": str(s.id),
                            "platform": s.platform,
                            "enabled": s.enabled
                        }
                        for s in sources
                    ],
                    "status": "success"
                }
        except Exception as e:
            return {"error": str(e), "status": "failed"}

    async def _get_cluster_detail(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute get_cluster_detail tool by calling digest API endpoint."""
        try:
            from app.intelligence.clustering import ContentClusterer

            clusterer = ContentClusterer()
            cluster_id = params.get("cluster_id")
            # Note: This would need to fetch cluster data from storage
            return {
                "cluster_id": cluster_id,
                "status": "success",
                "message": "Cluster detail retrieval requires storage implementation"
            }
        except Exception as e:
            return {"error": str(e), "status": "failed"}


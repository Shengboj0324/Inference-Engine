# MCP (Model Context Protocol) Integration

Social Media Radar implements the Model Context Protocol, allowing AI assistants and other MCP-compatible clients to interact with your intelligence desk.

## What is MCP?

The Model Context Protocol is a standard for exposing tools and resources that AI models can use. It allows your AI assistant to:

- Query your personalized content digest
- Search through your content backlog
- Configure new content sources
- Get detailed information about topics

## Available Tools

### 1. get_daily_digest

Get a personalized daily digest of content from your configured sources.

**Parameters:**
- `topics` (optional): Filter by specific topics
- `since_hours` (default: 24): Hours to look back
- `max_clusters` (default: 20): Maximum number of topic clusters
- `platforms` (optional): Filter by specific platforms

**Example:**
```json
{
  "topics": ["AI", "technology"],
  "since_hours": 24,
  "max_clusters": 15,
  "platforms": ["reddit", "youtube"]
}
```

**Response:**
```json
{
  "generated_at": "2024-01-15T10:00:00Z",
  "period_start": "2024-01-14T10:00:00Z",
  "period_end": "2024-01-15T10:00:00Z",
  "clusters": [
    {
      "topic": "New AI Model Release",
      "summary": "Multiple sources report...",
      "items": [...],
      "platforms_represented": ["reddit", "youtube"],
      "relevance_score": 0.92
    }
  ],
  "total_items": 156
}
```

### 2. search_content

Search through your content backlog using natural language queries.

**Parameters:**
- `query` (required): Search query
- `platforms` (optional): Filter by platforms
- `since_hours` (optional): Hours to look back
- `limit` (default: 50): Maximum results

**Example:**
```json
{
  "query": "machine learning breakthroughs",
  "platforms": ["reddit", "youtube"],
  "limit": 20
}
```

### 3. configure_source

Configure a new content source or update existing configuration.

**Parameters:**
- `platform` (required): Platform name (reddit, youtube, rss, etc.)
- `credentials` (required): Platform-specific credentials
- `settings` (optional): Platform-specific settings

**Example for Reddit:**
```json
{
  "platform": "reddit",
  "credentials": {
    "client_id": "your_client_id",
    "client_secret": "your_client_secret",
    "refresh_token": "your_refresh_token"
  },
  "settings": {
    "max_subreddits": 50
  }
}
```

### 4. list_sources

List all configured content sources.

**Parameters:** None

**Response:**
```json
{
  "sources": [
    {
      "platform": "reddit",
      "enabled": true,
      "connection_status": "connected",
      "feeds_count": 42
    },
    {
      "platform": "youtube",
      "enabled": true,
      "connection_status": "connected",
      "feeds_count": 28
    }
  ]
}
```

### 5. get_cluster_detail

Get detailed information about a specific content cluster.

**Parameters:**
- `cluster_id` (required): Cluster UUID

**Response:**
```json
{
  "cluster": {
    "id": "uuid",
    "topic": "AI Safety Debate",
    "summary": "Extended summary...",
    "items": [...],
    "perspective_summary": "Reddit users focus on..., while YouTube creators emphasize..."
  }
}
```

## Using MCP with AI Assistants

### Claude Desktop

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "social-media-radar": {
      "command": "python",
      "args": ["-m", "app.mcp_server.server"],
      "env": {
        "API_URL": "http://localhost:8000"
      }
    }
  }
}
```

### VS Code with Continue

Add to your Continue configuration:

```json
{
  "tools": [
    {
      "type": "mcp",
      "name": "social-media-radar",
      "url": "http://localhost:8001"
    }
  ]
}
```

## Example Workflows

### Morning Briefing

Ask your AI assistant:
> "Get my daily digest for the last 24 hours, focusing on AI and technology topics"

The assistant will use the `get_daily_digest` tool and present you with a formatted briefing.

### Deep Research

Ask your AI assistant:
> "Search my content for information about transformer architectures from the last week"

The assistant will use `search_content` and summarize the findings.

### Source Management

Ask your AI assistant:
> "Add my YouTube account as a content source"

The assistant will guide you through the OAuth flow and use `configure_source`.

## Security Considerations

### Authentication

MCP tools require authentication. The server validates:
- User identity via JWT tokens
- API key for MCP client
- Rate limiting per user

### Credential Handling

- Credentials are encrypted at rest
- Never logged or exposed in responses
- Transmitted over HTTPS only
- Rotated regularly

### Data Privacy

- Each user's data is isolated
- No cross-user data access
- Audit logging for all operations
- GDPR-compliant export/delete

## Development

### Running the MCP Server

```bash
# Start the MCP server
python -m app.mcp_server.server

# Or with Docker
docker-compose up mcp-server
```

### Testing MCP Tools

```python
from app.mcp_server.server import MCPServer

server = MCPServer()

# List available tools
tools = server.list_tools()

# Execute a tool
result = await server.execute_tool(
    "get_daily_digest",
    {"since_hours": 24, "max_clusters": 10}
)
```

## Troubleshooting

### Connection Issues

- Verify API server is running
- Check MCP server logs
- Validate authentication tokens
- Ensure network connectivity

### Tool Execution Errors

- Check tool parameters match schema
- Verify user has configured sources
- Check API rate limits
- Review server logs for details

## Future Enhancements

- [ ] Real-time streaming updates
- [ ] Custom tool creation
- [ ] Multi-user collaboration
- [ ] Advanced filtering and sorting
- [ ] Export to various formats


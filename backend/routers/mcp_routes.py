"""
routers/mcp_routes.py — MCP status and prompt endpoints.

Routes:
  GET /mcp-status  — live MCP server info (tools, resources, prompts)
  GET /mcp-prompt  — expand an MCP prompt template
"""

from fastapi import APIRouter, HTTPException

from backend.dependencies import services
from backend.mcp_server import mcp

router = APIRouter(tags=["mcp"])


@router.get("/mcp-status")
async def mcp_status():
    """
    Return live status of the connected MCP server.

    Shows all discovered tools, resources, and prompts.
    Used by the UI to display the MCP server info panel.
    """
    if services.mcp_client is None:
        return {
            "connected": False,
            "error":     "MCP client not initialized (no API key configured)",
            "tools":     [], "resources": [], "prompts": [],
        }
    return await services.mcp_client.get_server_info()


@router.get("/mcp-prompt")
async def get_mcp_prompt(name: str, arguments: str = "{}"):
    """
    Expand an MCP prompt template and return the resulting text.

    Called by the frontend when a user selects a /prompt from the autocomplete.
    """
    import json as _json
    try:
        args   = _json.loads(arguments)
        result = await mcp.get_prompt(name, args)
        text   = ""
        for msg in result.messages:
            if hasattr(msg.content, "text"):
                text = msg.content.text
                break
        return {"name": name, "text": text}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prompt error: {e}")

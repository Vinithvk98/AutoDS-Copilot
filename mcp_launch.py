"""Path-robust entry point for the AutoDS MCP server.

MCP clients (Claude Desktop, etc.) launch this file directly. It makes the
`autods` package importable no matter what working directory the client uses,
then starts the stdio server.

    python3 mcp_launch.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autods.mcp_server import main  # noqa: E402

if __name__ == "__main__":
    main()

import json
import os
import sys
from pathlib import Path
from typing import Dict

# Executed with the configured HEPTAPOD interpreter, which supplies the MCP SDK.
from mcp import Client, StdioServerParameters
from mcp.server.mcpserver import MCPServer

from .gateway import TOOLS, InspectionGateway


def main() -> None:
    policy = json.loads(Path(sys.argv[1]).read_text())

    async def invoke(upstream: Dict, name: str, arguments: Dict) -> object:
        environment = dict(os.environ)
        environment.update(upstream['env'])
        parameters = StdioServerParameters(
            command=upstream['command'],
            args=upstream['args'],
            env=environment,
            cwd=upstream['cwd']
        )
        async with Client(parameters) as client:
            return await client.call_tool(name, arguments)

    gateway = InspectionGateway(policy, invoke)
    server = MCPServer("Bound HEPTAPOD inspection tools")

    async def preflight() -> str:
        return await gateway.call('VisualInspectionPreflight')

    async def inspect() -> str:
        return await gateway.call('InspectBoard')

    async def read_result() -> str:
        return await gateway.call('ReadInspectionResult')

    functions = dict(zip(TOOLS, (preflight, inspect, read_result)))
    for name in policy['tools']:
        server.tool(
            name=name,
            description=f"{name} for the user-selected board and authorized action. Inputs are fixed server-side."
        )(functions[name])

    server.run()


if __name__ == '__main__':
    main()

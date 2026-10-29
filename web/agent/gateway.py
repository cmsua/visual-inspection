import asyncio
import json
from typing import Awaitable, Callable, Dict


TOOLS = ('VisualInspectionPreflight', 'InspectBoard', 'ReadInspectionResult')


def decode_result(result: object) -> Dict:
    if result.is_error:
        raise ValueError("HEPTAPOD rejected the tool call.")

    for block in result.content:
        if getattr(block, 'type', None) == 'text':
            try:
                envelope = json.loads(block.text)
            except ValueError:
                continue

            if isinstance(envelope, dict) and envelope.get('schema_version') == 'visual-inspection-bridge/1':
                if envelope.get('status') != 'ok':
                    raise ValueError(json.dumps(envelope.get('error', envelope)))

                return envelope

    raise ValueError("HEPTAPOD returned no supported structured evidence.")


class InspectionGateway:
    def __init__(self, policy: Dict, invoke: Callable[[Dict, str, Dict], Awaitable[object]]) -> None:
        self.invoke = invoke
        self.policy = policy
        self.ready = False
        self.calls = set()
        self.result_path = policy.get('result_path')
        self.lock = asyncio.Lock()

    async def call(self, name: str) -> str:
        async with self.lock:
            if name not in self.policy['tools'] or name in self.calls:
                raise ValueError("This tool is not authorized or has already run for this request.")
            if name == 'InspectBoard' and not self.ready:
                raise ValueError("Inspection requires a successful ready preflight in this request.")

            arguments = {'config_path': self.policy['config_path'], 'board_path': self.policy['board_path']}
            if name == 'InspectBoard':
                arguments['force'] = self.policy['action'] == 'refresh'
            elif name == 'ReadInspectionResult':
                if not self.result_path:
                    raise ValueError("No saved inspection snapshot was selected or produced.")

                arguments = {'result_path': self.result_path}

            self.calls.add(name)
            upstream = self.policy['upstream']
            result = await self.invoke(upstream, name, arguments)

            envelope = decode_result(result)
            payload = envelope['result']
            if name == 'VisualInspectionPreflight':
                self.ready = payload.get('ready') is True
            elif name == 'InspectBoard':
                self.result_path = payload['result_path']

            return json.dumps(envelope, allow_nan=False)

"""Verifier-owned HTTP bridge to the existing observed SQL/ssgrep shims.

The native solver has neither database passwords nor filesystem access to the index.
Only bounded search/show and its assigned database/role cross this boundary. The
server's environment is the isolated actor kit, never request-controlled input.
Native tool output is capped at the observer blob bound per stream; encoded HTTP bodies
have a separate hard bound. Capture failures expose only status, never a partial answer.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .bounded_capture import capture_process, decode_text
from .domain_lane import write_private
from .lab_product import REPO_ROOT
from .lab_session import ActorKit
from .observer import MAX_BLOB_BYTES

MAX_BODY = 128_000
BRIDGE_STDOUT_LIMIT = MAX_BLOB_BYTES
BRIDGE_STDERR_LIMIT = MAX_BLOB_BYTES
# JSON can expand each source byte to a six-byte Unicode/control escape.
MAX_RESPONSE_BYTES = 6 * (BRIDGE_STDOUT_LIMIT + BRIDGE_STDERR_LIMIT) + 4096
REF = re.compile(r"[a-z][a-z0-9_-]*:[\w-]+:ep:\d+")
TRUSTED_SHIM = (
    f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); "
    "from eval.accounting_airline.observer import shim_main; "
    "raise SystemExit(shim_main(sys.argv[1], sys.argv[2:], group_owner=True))"
)


def validate_request(tool: str, args: list[str], *, database: str, writer: bool) -> str:
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise ValueError("argv must be strings")
    if tool == "sql":
        allowed = [
            ["--role", role, "--database", database]
            for role in (("reader", "writer") if writer else ("reader",))
        ]
        if args not in allowed:
            raise ValueError("SQL restricted to the assigned database and role")
    elif tool == "ssgrep":
        json_lookup = len(args) == 3 and args[0] in ("search", "show") and args[2] == "--json"
        readable_show = len(args) == 2 and args[0] == "show"
        if not (json_lookup or readable_show):
            raise ValueError("only search <query> --json and show <ref> [--json] are allowed")
        if not args[1] or len(args[1]) > 4000:
            raise ValueError("empty or oversized lookup")
        if args[0] == "show" and not REF.fullmatch(args[1]):
            raise ValueError("show requires a returned episode ref")
    else:
        raise ValueError("tool not allowed")
    return tool


class Bridge:
    def __init__(self, kit: ActorKit, database: str, cwd: Path):
        self.kit, self.database, self.cwd = kit, database, cwd
        self.writer = False
        self.attempts: list[dict[str, Any]] = []
        self.stop = threading.Event()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def setup(self) -> None:
                super().setup()
                # A vanished native client cannot indefinitely block owner-group cleanup.
                self.connection.settimeout(10)

            def log_message(self, format: str, *_args: Any) -> None:
                pass

            def do_POST(self) -> None:
                attempt: dict[str, Any] | None = None
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if self.path != "/tool" or not 0 < size <= MAX_BODY:
                        raise ValueError("invalid request boundary")
                    body = json.loads(self.rfile.read(size))
                    tool, args = body["tool"], body["args"]
                    validate_request(tool, args, database=outer.database, writer=outer.writer)
                    stdin = body.get("stdin", "")
                    if not isinstance(stdin, str) or (tool != "sql" and stdin):
                        raise ValueError("stdin only permitted for SQL")
                    attempt_dir = outer.kit.spool.parent / "bridge-attempts" / uuid.uuid4().hex
                    done = capture_process(
                        [sys.executable, "-I", "-c", TRUSTED_SHIM, tool, *args],
                        input_bytes=stdin.encode(),
                        env={**outer.kit.env, "LAB_CAPTURE_DIR": str(attempt_dir)},
                        cwd=outer.cwd,
                        timeout=180,
                        stdout_limit=BRIDGE_STDOUT_LIMIT,
                        stderr_limit=BRIDGE_STDERR_LIMIT,
                        output_dir=attempt_dir,
                        failure_marker=attempt_dir / "tool-capture-failed",
                        cancel_event=outer.stop,
                    )
                    attempt = {**done.metadata(), "artifact_dir": str(attempt_dir), "tool": tool}
                    tool_capture = attempt_dir / "tool-capture.json"
                    if tool_capture.exists():
                        with tool_capture.open("rb") as handle:
                            raw = handle.read(4097)
                        if len(raw) > 4096:
                            raise ValueError("oversized trusted capture metadata")
                        attempt["tool_capture"] = json.loads(raw)
                    data = {
                        **done.metadata(),
                        "returncode": done.returncode,
                        "stdout": decode_text(done.stdout) if done.status == "completed" else "",
                        "stderr": decode_text(done.stderr)
                        if done.status == "completed"
                        else f"restricted tool bridge: {done.status}",
                    }
                    code = 200
                except (ValueError, KeyError, TypeError) as exc:
                    data, code = {"error": str(exc)}, 400
                except Exception as exc:
                    data, code = {"error": f"bridge execution: {type(exc).__name__}"}, 500
                payload = json.dumps(data).encode()
                if len(payload) > MAX_RESPONSE_BYTES:
                    data = {
                        "returncode": data.get("returncode"),
                        "stdout": "",
                        "stderr": "restricted tool bridge: output_limit",
                        "capture_status": "output_limit",
                        "output_limit": True,
                    }
                    if attempt is not None:
                        attempt.update(
                            output_limit=True,
                            capture_status="output_limit",
                            output_limit_streams=["http_response"],
                        )
                    payload = json.dumps(data).encode()
                    code = 200
                if attempt is not None:
                    attempt["http_response_bytes"] = len(payload)
                    write_private(Path(attempt["artifact_dir"]) / "result.json", attempt)
                    outer.attempts.append(attempt)
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = False
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> Bridge:
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.stop.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def install_clients(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        endpoint = f"http://127.0.0.1:{self.server.server_port}/tool"
        for tool in ("sql", "ssgrep"):
            path = directory / tool
            path.write_text(
                f"#!{sys.executable}\n"
                "import json,sys,urllib.request\n"
                f"body={{'tool':{tool!r},'args':sys.argv[1:],"
                f"'stdin':sys.stdin.buffer.read({MAX_BODY + 1}).decode() "
                f"if {tool!r} == 'sql' else ''}}\n"
                "payload=json.dumps(body).encode()\n"
                f"if len(payload)>{MAX_BODY}:\n"
                " print('restricted tool bridge: invalid request boundary',"
                "file=sys.stderr);sys.exit(1)\n"
                f"request=urllib.request.Request({endpoint!r},payload,"
                "{'Content-Type':'application/json'})\n"
                "try:\n"
                " with urllib.request.urlopen(request,timeout=190) as response:\n"
                f"  payload=response.read({MAX_RESPONSE_BYTES + 1})\n"
                f"  if len(payload)>{MAX_RESPONSE_BYTES}: raise ValueError('bridge output_limit')\n"
                "  result=json.loads(payload)\n"
                "except Exception as exc:\n"
                " print('restricted tool bridge: '+str(exc),file=sys.stderr);sys.exit(1)\n"
                "if result.get('capture_status','completed')!='completed':\n"
                " sys.stderr.write(result.get('stderr','restricted tool bridge: failed capture')); "
                "sys.exit(result['returncode'] or 1)\n"
                "sys.stdout.write(result['stdout']);sys.stderr.write(result['stderr']);sys.exit(result['returncode'])\n"
            )
            path.chmod(0o700)

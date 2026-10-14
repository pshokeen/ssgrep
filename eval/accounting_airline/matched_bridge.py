"""Matched bridge: identical database names and server-enforced history capability."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .live_bridge import MAX_BODY, TRUSTED_SHIM, Bridge, validate_request


def request_args(tool, args, *, database, history):
    """Validate without rewriting argv: native and observed calls must remain identical."""
    if tool == "ssgrep" and not history:
        raise ValueError("matched.history_disabled: history capability unavailable")
    validate_request(tool, args, database=database, writer=True)
    return list(args)


class MatchedBridge(Bridge):
    def __init__(self, kit, database, cwd, *, history):
        self.kit, self.database, self.cwd = kit, database, cwd
        self.writer = False
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                pass

            def do_POST(self):
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if self.path != "/tool" or not 0 < size <= MAX_BODY:
                        raise ValueError("invalid request boundary")
                    body = json.loads(self.rfile.read(size))
                    tool = body["tool"]
                    args = request_args(tool, body["args"], database=database, history=history)
                    validate_request(tool, args, database=database, writer=outer.writer)
                    stdin = body.get("stdin", "")
                    if not isinstance(stdin, str) or (tool != "sql" and stdin):
                        raise ValueError("stdin only permitted for SQL")
                    done = subprocess.run(
                        [sys.executable, "-I", "-c", TRUSTED_SHIM, tool, *args],
                        input=stdin,
                        text=True,
                        capture_output=True,
                        env=kit.env,
                        cwd=cwd,
                        timeout=180,
                        check=False,
                    )
                    data = {
                        "returncode": done.returncode,
                        "stdout": done.stdout,
                        "stderr": done.stderr,
                    }
                    code = 200
                except (ValueError, KeyError, TypeError) as exc:
                    data, code = {"error": str(exc)}, 400
                except Exception as exc:
                    data, code = {"error": f"bridge execution: {type(exc).__name__}"}, 500
                payload = json.dumps(data).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

"""Verifier-owned HTTP bridge to the existing observed SQL/ssgrep shims.

The native solver has neither database passwords nor filesystem access to the index.
Only bounded search/show and its assigned database/role cross this boundary. The
server's environment is the isolated actor kit, never request-controlled input.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .lab_product import REPO_ROOT
from .lab_session import ActorKit

MAX_BODY = 128_000
REF = re.compile(r"[a-z][a-z0-9_-]*:[\w-]+:ep:\d+")
TRUSTED_SHIM = (
    f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); "
    "from eval.accounting_airline.observer import shim_main; "
    "raise SystemExit(shim_main(sys.argv[1], sys.argv[2:]))"
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
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *_args: Any) -> None:
                pass

            def do_POST(self) -> None:
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
                    done = subprocess.run(
                        [sys.executable, "-I", "-c", TRUSTED_SHIM, tool, *args],
                        input=stdin,
                        text=True,
                        capture_output=True,
                        env=outer.kit.env,
                        cwd=outer.cwd,
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

    def __enter__(self) -> Bridge:
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
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
                f"'stdin':sys.stdin.read() if {tool!r} == 'sql' else ''}}\n"
                f"request=urllib.request.Request({endpoint!r},json.dumps(body).encode(),"
                "{'Content-Type':'application/json'})\n"
                "try:\n"
                " with urllib.request.urlopen(request,timeout=190) as response:\n"
                "  result=json.load(response)\n"
                "except Exception as exc:\n"
                " print('restricted tool bridge: '+str(exc),file=sys.stderr);sys.exit(1)\n"
                "sys.stdout.write(result['stdout']);sys.stderr.write(result['stderr']);sys.exit(result['returncode'])\n"
            )
            path.chmod(0o700)

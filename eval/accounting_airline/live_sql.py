"""libpq SQL-only backend for the existing observer shim's psql-shaped argv.

libpq sends SQL to PostgreSQL; unlike psql -f, it has no client shell/include/copy
metacommands. This adapter deliberately uses libpq already installed with psql,
not a new network dependency, and emits the shim's established tabular format.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


def library_path(psql: str) -> Path:
    prefix = Path(psql).resolve().parent.parent
    for directory in (prefix / "lib", prefix / "lib/postgresql@17"):
        for name in ("libpq.5.dylib", "libpq.dylib", "libpq.so.5", "libpq.so"):
            candidate = directory / name
            if candidate.is_file():
                return candidate
    raise RuntimeError(f"libpq prerequisite missing beside psql prefix {prefix}")


def _lib(path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(path)
    pointer = ctypes.c_void_p
    signatures = {
        "PQconnectdb": (pointer, [ctypes.c_char_p]),
        "PQstatus": (ctypes.c_int, [pointer]),
        "PQerrorMessage": (ctypes.c_char_p, [pointer]),
        "PQsendQuery": (ctypes.c_int, [pointer, ctypes.c_char_p]),
        "PQgetResult": (pointer, [pointer]),
        "PQresultStatus": (ctypes.c_int, [pointer]),
        "PQresultErrorMessage": (ctypes.c_char_p, [pointer]),
        "PQresultErrorField": (ctypes.c_char_p, [pointer, ctypes.c_int]),
        "PQntuples": (ctypes.c_int, [pointer]),
        "PQnfields": (ctypes.c_int, [pointer]),
        "PQfname": (ctypes.c_char_p, [pointer, ctypes.c_int]),
        "PQgetvalue": (ctypes.c_char_p, [pointer, ctypes.c_int, ctypes.c_int]),
        "PQclear": (None, [pointer]),
        "PQfinish": (None, [pointer]),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(lib, name)
        function.restype, function.argtypes = restype, argtypes
    return lib


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    lib = _lib(os.environ["AIRLINE_LIBPQ_PATH"])

    def flag(name: str) -> str:
        return args[args.index(name) + 1]

    values = {
        "host": flag("-h"),
        "port": flag("-p"),
        "user": flag("-U"),
        "dbname": flag("-d"),
        "passfile": os.environ["PGPASSFILE"],
        "connect_timeout": "10",
        "application_name": "airline-live-sql",
    }

    def quoted(value: str) -> str:
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"

    conninfo = " ".join(f"{key}={quoted(value)}" for key, value in values.items())
    conn = lib.PQconnectdb(conninfo.encode())
    if not conn or lib.PQstatus(conn) != 0:
        print("SQL connection prerequisite failed", file=sys.stderr)
        if conn:
            lib.PQfinish(conn)
        return 1
    result = None
    try:
        sql = sys.stdin.read()
        if "\x00" in sql:
            print("ERROR: 42601: SQL contains NUL", file=sys.stderr)
            return 1
        if lib.PQsendQuery(conn, sql.encode()) != 1:
            print("SQL protocol could not send query", file=sys.stderr)
            return 1
        failed = False
        while result := lib.PQgetResult(conn):
            status = lib.PQresultStatus(result)
            if status in (3, 4, 8):
                print(
                    "ERROR: 0A000: streaming COPY is not supported by the SQL tool", file=sys.stderr
                )
                return 1
            if status not in (1, 2):
                state = (lib.PQresultErrorField(result, ord("C")) or b"?????").decode()
                detail = (lib.PQresultErrorMessage(result) or b"SQL failed").decode(
                    errors="replace"
                )
                print(f"ERROR: {state}: {detail}", file=sys.stderr)
                failed = True
            if status == 2:
                columns = lib.PQnfields(result)
                print("\t".join(lib.PQfname(result, col).decode() for col in range(columns)))
                for row in range(lib.PQntuples(result)):
                    print(
                        "\t".join(
                            lib.PQgetvalue(result, row, col).decode() for col in range(columns)
                        )
                    )
            lib.PQclear(result)
            result = None
        return int(failed)
    finally:
        if result:
            lib.PQclear(result)
        lib.PQfinish(conn)


if __name__ == "__main__":
    raise SystemExit(main())

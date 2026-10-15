"""Reuse existing RootLedger evaluations through immutable per-root blob handles."""

from __future__ import annotations

from collections import defaultdict

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.collector_types import CollectError
from ssgrep.accounting.plain_codec import from_plain, to_plain
from ssgrep.accounting.receipts import ReceiptRecord
from ssgrep.accounting.valuation import Leaf

BLOB = "root-evaluations"


def restore_cache(profile, ledger, blobs):
    raw = store.read_blob(profile, blobs.get(BLOB))
    if raw is None:
        return
    if not isinstance(raw, dict) or raw.get("schema") != 1:
        raise CollectError("state_corrupt", "Root evaluation cache is invalid.")
    if raw.get("methodology") != ledger.methodology.hash:
        return  # A method namespace can never borrow another method's canonical rows.
    try:
        ledger._root_cache_files = dict(raw["roots"])
        ledger._root_cache_keys = {}
        for root, filename in raw["roots"].items():
            rows = store.read_blob(profile, filename)
            if not isinstance(rows, list):
                raise ValueError("invalid root evaluation rows")
            for row in rows:
                keys = tuple(row["keys"])
                leaves = tuple(from_plain(Leaf, leaf) for leaf in row["leaves"])
                receipts = {
                    name: from_plain(ReceiptRecord, value)
                    for name, value in row["receipts"].items()
                }
                ledger._eval_cache[(root, keys)] = (leaves, receipts)
                ledger._root_cache_keys.setdefault(root, []).append(keys)
    except (KeyError, ValueError, TypeError) as error:
        raise CollectError("state_corrupt", "Root evaluation cache rows are invalid.") from error


def save_cache(profile, ledger, blobs):
    grouped = defaultdict(list)
    for root, keys in sorted(ledger._eval_cache):
        grouped[root].append(keys)
    roots = {}
    old_files = ledger._root_cache_files
    old_keys = ledger._root_cache_keys
    for root, keys in sorted(grouped.items()):
        if keys == old_keys.get(root) and root in old_files:
            roots[root] = old_files[root]
            continue
        rows = []
        for key in keys:
            leaves, receipts = ledger._eval_cache[(root, key)]
            rows.append(
                {"keys": list(key), "leaves": to_plain(leaves), "receipts": to_plain(receipts)}
            )
        roots[root] = store.write_blob(profile, "root-evaluation", rows)
    blobs[BLOB] = store.write_blob(
        profile,
        BLOB,
        {
            "schema": 1,
            "methodology": ledger.methodology.hash,
            "roots": roots,
        },
    )


def references(profile, filename):
    raw = store.read_blob(profile, filename)
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != 1
        or not isinstance(raw.get("roots"), dict)
    ):
        raise CollectError("state_corrupt", "Root evaluation cache manifest is invalid.")
    return set(raw["roots"].values())

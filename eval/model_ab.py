"""Settles open question 1: does potion-retrieval-32M (MTEB Retrieval
35.06) justify its larger model file over potion-base-8M (31.11)?

Builds two private indexes of the same real corpus scope -- one per model --
by monkeypatching ssgrep.embed's module attributes for the duration of each
build, and compares recall@10 / MRR on the same labelled query set. Never
edits src/ssgrep/embed.py; the patch is applied and reverted in-process.

potion-retrieval-32M encodes at 512 dimensions (measured here, not assumed --
potion-base-8M is 256), so both MODEL_ID and DIMENSION are patched together;
embed.get_model_info() is what indexer.py and search.py's query embedding
consult, and both read it as a live module attribute, not an import-time copy.
"""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path

from eval import harness, provenance
from ssgrep import embed, store, vectors

MODELS = {
    "potion-base-8M": "minishlab/potion-base-8M",  # current default, MTEB Retrieval 31.11
    "potion-retrieval-32M": "minishlab/potion-retrieval-32M",  # MTEB Retrieval 35.06
}

LABELS = {
    "potion-base-8M": "potion-base-8M (current shipped default, MTEB Retrieval 31.11)",
    "potion-retrieval-32M": "potion-retrieval-32M (MTEB Retrieval 35.06)",
}


def _probe_dimension(model_id: str) -> int:
    from model2vec import StaticModel

    model = StaticModel.from_pretrained(model_id)
    return int(model.encode(["probe"]).shape[1])


def _dimension_aware_validate_alignment(db_path: Path, vec_path: Path) -> tuple[bool, str]:
    """Same logic as vectors.validate_alignment, but reads embed.DIMENSION at
    call time instead of a hardcoded 256.

    FINDING (report, do not fix here -- vectors.py is not owned by this
    task): vectors.validate_alignment() hardcodes `256` for the vector row
    size. It is called from both indexer._needs_rebuild() and search.py's
    _open_ready_index() -- i.e. on both the index-build path and the query
    path -- so a non-256-dim model does not just fail to build; a query
    against an already-built non-256-dim index raises IndexNotReadyError
    (misaligned) even when it is perfectly consistent. This is the same class
    of bug as GenerationalStore.VECTOR_ROW_BYTES, in a second file.
    """
    import sqlite3

    dimension = embed.DIMENSION
    try:
        if not db_path.exists():
            return True, ""
        conn = sqlite3.connect(str(db_path))
        if not vec_path.exists():
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            conn.close()
            if chunk_count > 0:
                return False, "index.db has chunks but vectors.f32 is missing"
            return True, ""
        vec_size = vec_path.stat().st_size
        vec_row_count = vec_size // (dimension * 4)
        invalid_chunks = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE vec_row IS NULL OR vec_row >= ?",
            (vec_row_count,),
        ).fetchone()[0]
        if invalid_chunks > 0:
            conn.close()
            return False, f"index.db has {invalid_chunks} chunks with missing or invalid vec_row"
        chunk_rows = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT vec_row FROM chunks WHERE vec_row IS NOT NULL"
            ).fetchall()
        }
        valid_rows = set(range(vec_row_count))
        orphaned = valid_rows - chunk_rows
        conn.close()
        if orphaned:
            return False, f"vectors.f32 has {len(orphaned)} orphaned rows with no owning chunk"
        return True, ""
    except Exception as e:  # pragma: no cover - mirrors production's own catch-all
        return False, f"validation error: {e}"


def run_model_ab(
    *, main_session_boost: float, base_dir: Path | None = None, task: str = ""
) -> dict:
    base_dir = base_dir or Path(tempfile.mkdtemp(prefix="ssgrep-eval-model-ab-"))
    run_date = date.today().isoformat()
    original_model_id = embed.MODEL_ID
    original_dimension = embed.DIMENSION
    original_model_obj = embed._model
    original_vector_row_bytes = store.GenerationalStore.VECTOR_ROW_BYTES
    original_validate_alignment = vectors.validate_alignment

    out = {}
    try:
        vectors.validate_alignment = _dimension_aware_validate_alignment
        for label, model_id in MODELS.items():
            dimension = _probe_dimension(model_id)
            embed.MODEL_ID = model_id
            embed.DIMENSION = dimension
            embed._model = None  # force reload under the new MODEL_ID

            # FINDING (report, do not fix here -- store.py is not owned by this
            # task): GenerationalStore.VECTOR_ROW_BYTES is hardcoded to
            # `256 * 4`, i.e. it silently assumes potion-base-8M's dimension.
            # With an actually-larger model (potion-retrieval-32M is 512-dim,
            # measured, not assumed) commit_generation()'s consistency check
            # miscomputes the live row count and refuses to commit a
            # perfectly valid generation. The design's claim that a model swap
            # is "a one-line change plus a re-vectorize" is false as long as
            # this constant exists; it is at least a two-file change. Patched
            # here, in-process, only for the duration of this experiment.
            store.GenerationalStore.VECTOR_ROW_BYTES = dimension * 4

            index_dir = base_dir / label
            results, summary, stats = harness.run(
                index_dir, main_session_boost=main_session_boost, rebuild=True
            )
            db_path, _vec_path = harness._index_paths_for(index_dir)
            out[label] = {
                "model_id": model_id,
                "dimension": dimension,
                "episode_count": stats.episode_count,
                "chunk_count": stats.chunk_count,
                "summary": summary,
                "provenance": provenance.build_provenance(
                    date=run_date,
                    task=task,
                    label=LABELS.get(label, label),
                    stats=stats,
                    db_path=db_path,
                ),
            }
            print(f"=== {label} ({model_id}, dim={dimension}) ===")
            for key, agg in summary.items():
                if agg["n"] == 0:
                    continue
                print(
                    f"  {key:20s} n={agg['n']:3d}  "
                    f"recall@10={agg['recall_at_10']:.3f}  mrr={agg['mrr']:.3f}"
                )
    finally:
        embed.MODEL_ID = original_model_id
        embed.DIMENSION = original_dimension
        embed._model = original_model_obj
        store.GenerationalStore.VECTOR_ROW_BYTES = original_vector_row_bytes
        vectors.validate_alignment = original_validate_alignment

    return out


def main() -> None:
    import argparse

    from ssgrep import search as search_module

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-session-boost", type=float, default=search_module.MAIN_SESSION_BOOST
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--task",
        type=str,
        default="Re-run model A/B (potion-base-8M vs potion-retrieval-32M) after the "
        "chunk-dedup fix, at the shipped default boost (p6-abrerun)",
    )
    args = parser.parse_args()

    result = run_model_ab(main_session_boost=args.main_session_boost, task=args.task)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()

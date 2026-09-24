"""Per-runtime emitters converting generated sessions into adapter-native formats.

Each emitter converts ``eval/datasetgen/sessions.parquet`` rows for one runtime
into files the real ``ssgrep`` adapter for that runtime ingests unchanged, so
the benchmark corpus is consumed by pointing the documented env overrides at
the emitted directories and running the real indexer. Emitter contracts live in
``eval/datasetgen/adapter_formats.md`` section 10.
"""

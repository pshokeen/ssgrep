"""Opt-in local accounting of ssgrep lookups and modeled retrieval benefit.

Evidence flows one way: capture (journal) and raw runtime readers produce
immutable evidence; the reducer derives receipts and snapshots; CLI, HTTP, and
export read snapshots and never trigger collection. See the
``add-savings-ledger`` OpenSpec change for the normative contracts.
"""

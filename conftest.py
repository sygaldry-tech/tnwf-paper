"""
conftest.py — test infrastructure for tnwf.

Tiered markers (mirrors the research prototype convention):
  needle     — quick smoke tests (<30s)
  medium     — pipeline tests (30s-5min)

Usage:
  pytest                                 # all tests
  pytest -m needle                       # only needle
  pytest -m "needle or medium"           # both
"""
from __future__ import annotations

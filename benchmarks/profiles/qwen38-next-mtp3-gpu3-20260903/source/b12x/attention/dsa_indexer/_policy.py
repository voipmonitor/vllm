"""Typed component policy for DSA indexer planning."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import DSA_INDEXER, BackendConfig, make_fixed_backend_policy


@dataclass(frozen=True, kw_only=True)
class DsaIndexerQuery:
    source_layout: str
    mode: str
    dtype: str
    kv_dtype: str
    num_q_heads: int
    num_idx_heads: int
    max_q_rows: int
    max_k_rows: int
    top_k: int
    page_size: int
    score_mode: str
    shared_page_table: bool


DSA_INDEXER_POLICY = make_fixed_backend_policy(
    component_id=DSA_INDEXER,
    query_type=DsaIndexerQuery,
    backend="native",
)
DsaIndexerConfig = BackendConfig


__all__ = ["DSA_INDEXER_POLICY", "DsaIndexerConfig", "DsaIndexerQuery"]

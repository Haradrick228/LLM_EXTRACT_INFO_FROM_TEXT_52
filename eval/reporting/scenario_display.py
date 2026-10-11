"""Human-readable labels for B3 matrix scenario rows (dense base_id → model name)."""
from __future__ import annotations

from typing import Any, Dict

DENSE_ALIASES = {
    "full_mpnet": "mpnet",
    "full_bge_m3": "BGE-M3",
    "full_frida": "FRIDA",
    "full_e5_large_instruct": "E5-large-instruct",
    "full_jina_embeddings_v3": "Jina-embeddings-v3",
    "full_default_clean": "mpnet (legacy default index)",
    "full_user_bge_m3": "USER-bge-m3",
}


def embed_short(embed_model: str) -> str:
    return embed_model.rsplit("/", 1)[-1] if embed_model else "?"


def scenario_display_label(sid: str, summary: Dict[str, Any]) -> str:
    chroma = (summary.get("scenario_chroma") or {}).get(sid) or {}
    embed = str(chroma.get("embed_model") or "")
    coll = str(chroma.get("chroma_collection_resolved") or chroma.get("chroma_collection_yaml") or "")
    tag = DENSE_ALIASES.get(sid, sid)
    if sid == "full_default_clean":
        model_short = embed_short(embed) if embed else tag
        return f"{tag}: {model_short} · Chroma «{coll}»"
    if not embed:
        return tag
    model_short = embed_short(embed)
    if model_short == tag:
        return tag
    return f"{tag} ({model_short})"

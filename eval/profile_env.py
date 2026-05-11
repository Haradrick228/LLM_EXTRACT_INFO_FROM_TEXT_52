"""Backward-compatible re-exports; implementation lives in common.rag_profile_env."""
from common.rag_profile_env import (  # noqa: F401
    apply_profile,
    apply_profile_by_rag_profile_env,
    list_profile_names,
    load_profiles,
    profiles_path,
)

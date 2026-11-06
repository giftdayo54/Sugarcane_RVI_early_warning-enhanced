"""
src/secrets_util.py

Robust lookup of Sentinel Hub credentials. Streamlit's `st.secrets.get("SH_CLIENT_ID")` only matches an exact
top-level key, so credentials pasted into Streamlit Cloud *Settings -> Secrets* under a different name or inside a
[section] were silently ignored. This resolver looks, in order, at:

  1. st.secrets, top level  - exact key, then common aliases (case-insensitive)
  2. st.secrets, inside any [section] (e.g. [sentinelhub], [cdse], [copernicus]) - same aliases
  3. environment variables  - SH_CLIENT_ID, SH_CLIENT_SECRET, ... (same aliases)

Blank values and obvious placeholders ("your-client-id", "<...>") are ignored. Only key NAMES are ever reported for
diagnostics - never values.
"""
from __future__ import annotations

import os

PLACEHOLDER_PREFIXES = ("your", "<", "xxx", "changeme", "replace", "put_", "put-", "example")

ALIASES: dict[str, list[str]] = {
    "SH_CLIENT_ID": ["SH_CLIENT_ID", "SENTINELHUB_CLIENT_ID", "SENTINEL_HUB_CLIENT_ID", "SENTINELHUB_ID", "CDSE_CLIENT_ID",
                     "COPERNICUS_CLIENT_ID", "CLIENT_ID", "SH_ID"],
    "SH_CLIENT_SECRET": ["SH_CLIENT_SECRET", "SENTINELHUB_CLIENT_SECRET", "SENTINEL_HUB_CLIENT_SECRET", "SENTINELHUB_SECRET",
                         "CDSE_CLIENT_SECRET", "COPERNICUS_CLIENT_SECRET", "CLIENT_SECRET", "SH_SECRET"],
    "SH_BASE_URL": ["SH_BASE_URL", "SENTINELHUB_BASE_URL", "BASE_URL", "SH_URL"],
    "SH_TOKEN_URL": ["SH_TOKEN_URL", "SENTINELHUB_TOKEN_URL", "TOKEN_URL"],
}


def _usable(value) -> bool:
    if value is None or not isinstance(value, (str, int, float)):
        return False
    s = str(value).strip()
    return bool(s) and not s.lower().startswith(PLACEHOLDER_PREFIXES)


def _secrets_mapping():
    """st.secrets as a plain dict-like, or {} when there is no secrets file / not running under Streamlit."""
    try:
        import streamlit as st
        return {k: st.secrets[k] for k in st.secrets.keys()}
    except Exception:
        return {}


def _as_dict(obj):
    try:
        return {k: obj[k] for k in obj.keys()}
    except Exception:
        return None


def find_secret(key: str, default: str = "") -> tuple[str, str]:
    """Return (value, source_description). Source is '' when nothing usable was found (value == default)."""
    names = [n.upper() for n in ALIASES.get(key, [key])]
    top = _secrets_mapping()
    upper_top = {str(k).upper(): k for k in top}
    for n in names:
        if n in upper_top and _usable(top[upper_top[n]]):
            return str(top[upper_top[n]]).strip(), f"Streamlit secrets ({upper_top[n]})"
    for sec_name, sec in top.items():
        sec_d = _as_dict(sec)
        if sec_d is None:
            continue
        upper_sec = {str(k).upper(): k for k in sec_d}
        for n in names:
            if n in upper_sec and _usable(sec_d[upper_sec[n]]):
                return str(sec_d[upper_sec[n]]).strip(), f"Streamlit secrets ([{sec_name}] {upper_sec[n]})"
    for n in names:
        if _usable(os.environ.get(n)):
            return os.environ[n].strip(), f"environment variable {n}"
    return default, ""


def get_secret(key: str, default: str = "") -> str:
    return find_secret(key, default)[0]


def diagnostics() -> dict:
    """Names only (no values): what the app can see, to explain 'my secrets aren't loading'."""
    top = _secrets_mapping()
    sections = {}
    flat = []
    for k, v in top.items():
        d = _as_dict(v)
        if d is not None:
            sections[str(k)] = sorted(str(x) for x in d)
        else:
            flat.append(str(k))
    return {"top_level_keys": sorted(flat), "sections": sections,
            "found": {k: (find_secret(k)[1] or "not found") for k in ("SH_CLIENT_ID", "SH_CLIENT_SECRET")},
            "env_vars_set": [n for n in ("SH_CLIENT_ID", "SH_CLIENT_SECRET") if _usable(os.environ.get(n))]}

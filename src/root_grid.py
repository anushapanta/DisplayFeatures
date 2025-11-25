# src/root_grid.py
import re
from typing import Dict, Tuple, Optional, List
import numpy as np
import uproot

# Regexes to pull ieta/iphi from names (edit if you know your pattern)
IETA_IPHI_PATTERNS = [
    re.compile(r".*?\bieta\s*_?(-?\d+)\b.*?\biphi\s*_?(\d+)\b", re.IGNORECASE),
    re.compile(r".*?\bieta\s*_?(-?\d+)\b.*?\bphi\s*_?(\d+)\b", re.IGNORECASE),
    re.compile(r".*?ieta(-?\d+).*?iphi(\d+).*?", re.IGNORECASE),
    re.compile(r".*?ieta(-?\d+).*?phi(\d+).*?", re.IGNORECASE),
]

def try_extract_ieta_iphi(name: str) -> Optional[Tuple[int, int]]:
    for pat in IETA_IPHI_PATTERNS:
        m = pat.match(name)
        if m:
            try:
                return int(m.group(1)), int(m.group(2))
            except Exception:
                pass
    return None

def list_th1_keys(root_path: str) -> List[str]:
    """Return a flat list of all uproot keys that look like TH1-like (have to_numpy)."""
    keys = []
    with uproot.open(root_path) as f:
        for k in f.keys(recursive=True):
            obj = f[k]
            if hasattr(obj, "to_numpy"):
                key_str = k.decode() if isinstance(k, bytes) else str(k)
                keys.append(key_str)
    return keys

def index_histograms_by_name(root_path: str) -> Dict[Tuple[int, int], str]:
    """Try mapping by parsing ieta/iphi out of names."""
    mapping = {}
    with uproot.open(root_path) as f:
        for k in f.keys(recursive=True):
            obj = f[k]
            if hasattr(obj, "to_numpy"):
                key_str = k.decode() if isinstance(k, bytes) else str(k)
                disp = key_str.split(";")[0]
                pos = try_extract_ieta_iphi(disp)
                if pos:
                    mapping[pos] = key_str
    return mapping

def index_histograms_fallback_by_position(
    root_path: str,
    n_ieta: int,
    n_iphi: int,
    ieta_min: int,
    iphi_min: int,
    order: str = "ieta-major",
) -> Dict[Tuple[int, int], str]:
    """
    Map TH1s by their order if names don't carry ieta/iphi.
    order:
      - "ieta-major": ieta outer loop, iphi inner (row-major by ieta)
      - "iphi-major": iphi outer loop, ieta inner (column-major by iphi)
    """
    keys = list_th1_keys(root_path)
    total_cells = n_ieta * n_iphi
    if not keys:
        return {}

    # Use the first total_cells histograms (or all if fewer)
    keys = keys[:total_cells]

    mapping = {}
    idx = 0
    if order == "ieta-major":
        for i in range(n_ieta):
            for j in range(n_iphi):
                if idx >= len(keys): break
                ieta = ieta_min + i
                iphi = iphi_min + j
                mapping[(ieta, iphi)] = keys[idx]
                idx += 1
    else:  # iphi-major
        for j in range(n_iphi):
            for i in range(n_ieta):
                if idx >= len(keys): break
                ieta = ieta_min + i
                iphi = iphi_min + j
                mapping[(ieta, iphi)] = keys[idx]
                idx += 1
    return mapping

def index_histograms(
    root_path: str,
    n_ieta: int,
    n_iphi: int,
    ieta_min: int,
    iphi_min: int,
    order: str = "ieta-major",
) -> Dict[Tuple[int, int], str]:
    """
    Try name-based mapping first; if nothing matched, fall back to position mapping.
    """
    by_name = index_histograms_by_name(root_path)
    if by_name:
        return by_name
    # fallback
    return index_histograms_fallback_by_position(
        root_path, n_ieta, n_iphi, ieta_min, iphi_min, order
    )

def read_histogram(root_path: str, uproot_key: str):
    with uproot.open(root_path) as f:
        h = f[uproot_key]
        counts, edges = h.to_numpy()
        return counts, edges

def stats_from_hist(counts: np.ndarray, edges: np.ndarray):
    entries = int(np.sum(counts))
    mean = None
    std = None
    if entries > 0:
        centers = 0.5 * (edges[:-1] + edges[1:])
        mean = float((centers * counts).sum() / entries)
        var = float((counts * (centers - mean) ** 2).sum() / entries)
        std = float(np.sqrt(var))
    return {"entries": entries, "mean": mean, "std": std}

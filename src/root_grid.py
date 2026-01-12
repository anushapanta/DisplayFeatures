import re
from typing import Dict, Tuple, Optional, List
import numpy as np
import uproot

# Try to parse ieta/iphi from TH1 names. Edit these if your naming differs.
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
    keys = []
    with uproot.open(root_path) as f:
        for k in f.keys(recursive=True):
            obj = f[k]
            if hasattr(obj, "to_numpy"):
                keys.append(k.decode() if isinstance(k, bytes) else str(k))
    return keys

def index_histograms_by_name(root_path: str) -> Dict[Tuple[int, int], str]:
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
    """Map TH1s by order if names don't encode ieta/iphi."""
    keys = list_th1_keys(root_path)
    total_cells = n_ieta * n_iphi
    if not keys:
        return {}
    keys = keys[:total_cells]

    mapping = {}
    idx = 0
    if order == "ieta-major":
        for i in range(n_ieta):
            for j in range(n_iphi):
                if idx >= len(keys): break
                mapping[(ieta_min + i, iphi_min + j)] = keys[idx]; idx += 1
    else:  # iphi-major
        for j in range(n_iphi):
            for i in range(n_ieta):
                if idx >= len(keys): break
                mapping[(ieta_min + i, iphi_min + j)] = keys[idx]; idx += 1
    return mapping

def index_histograms(
    root_path: str,
    n_ieta: int,
    n_iphi: int,
    ieta_min: int,
    iphi_min: int,
    order: str = "ieta-major",
) -> Dict[Tuple[int, int], str]:
    """Prefer name-based mapping; fall back to position mapping."""
    by_name = index_histograms_by_name(root_path)
    if by_name:
        return by_name
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
    info = {"entries": entries, "mean": None, "std": None, "maxbin": None}
    if entries > 0:
        centers = 0.5 * (edges[:-1] + edges[1:])
        mean = float((centers * counts).sum() / entries)
        var = float((counts * (centers - mean) ** 2).sum() / entries)
        info["mean"] = mean
        info["std"] = float(np.sqrt(var))
        info["maxbin"] = float(centers[np.argmax(counts)])
    return info

def metric_value(counts: np.ndarray, edges: np.ndarray, metric: str) -> float:
    metric = (metric or "mean").lower()
    s = stats_from_hist(counts, edges)
    if metric == "entries":
        return float(s["entries"])
    if metric == "std":
        return float(s["std"]) if s["std"] is not None else float("nan")
    if metric == "maxbin":
        return float(s["maxbin"]) if s["maxbin"] is not None else float("nan")
    # default mean
    return float(s["mean"]) if s["mean"] is not None else float("nan")

def compute_heatmap(
    root_path: str,
    mapping: Dict[Tuple[int, int], str],
    n_ieta: int,
    n_iphi: int,
    ieta_min: int,
    iphi_min: int,
    metric: str = "mean",
):
    """Return (grid 2D list, vmin, vmax) for the chosen metric."""
    grid = [[float("nan") for _ in range(n_iphi)] for _ in range(n_ieta)]
    vals = []
    with uproot.open(root_path) as f:
        for (ieta, iphi), key in mapping.items():
            y = ieta - ieta_min
            x = iphi - iphi_min
            if not (0 <= y < n_ieta and 0 <= x < n_iphi):
                continue
            obj = f[key]
            if not hasattr(obj, "to_numpy"):
                continue
            counts, edges = obj.to_numpy()
            value = metric_value(counts, edges, metric)
            grid[y][x] = value
            if np.isfinite(value):
                vals.append(value)
    if vals:
        vmin, vmax = float(np.nanpercentile(vals, 1)), float(np.nanpercentile(vals, 99))
        if vmin == vmax:
            vmax = vmin + 1e-12
    else:
        vmin, vmax = 0.0, 1.0

    # convert NaN to None for JSON
    for y in range(n_ieta):
        for x in range(n_iphi):
            if not np.isfinite(grid[y][x]):
                grid[y][x] = None
    return grid, vmin, vmax

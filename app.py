from flask import Flask, render_template, request, redirect, url_for, jsonify, abort
import os, uuid
from cachetools import LRUCache
from src.root_grid import (
    index_histograms,
    read_histogram,
    stats_from_hist,
    compute_heatmap,
)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# token -> per-upload state
STATE = {}  # { token: {"path", "index", "cache", "heatmap_cache", "n_ieta"...} }

DEFAULTS = dict(n_ieta=96, n_iphi=256, ieta_min=0, iphi_min=0, order="ieta-major")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("rootfile")
    if not f or f.filename == "":
        return redirect(url_for("index"))

    token = str(uuid.uuid4())
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{token}.root")
    f.save(save_path)

    # read grid settings (optional)
    n_ieta   = int(request.form.get("n_ieta") or DEFAULTS["n_ieta"])
    n_iphi   = int(request.form.get("n_iphi") or DEFAULTS["n_iphi"])
    ieta_min = int(request.form.get("ieta_min") or DEFAULTS["ieta_min"])
    iphi_min = int(request.form.get("iphi_min") or DEFAULTS["iphi_min"])
    order    = request.form.get("order") or DEFAULTS["order"]

    try:
        mapping = index_histograms(save_path, n_ieta, n_iphi, ieta_min, iphi_min, order)
    except Exception as e:
        return render_template("index.html", error=f"Failed to read ROOT file: {e}")

    STATE[token] = {
        "path": save_path,
        "index": mapping,
        "cache": LRUCache(maxsize=2048),
        "heatmap_cache": {},  # metric -> payload
        "n_ieta": n_ieta,
        "n_iphi": n_iphi,
        "ieta_min": ieta_min,
        "iphi_min": iphi_min,
        "order": order,
    }
    return redirect(url_for("grid", token=token))


def _get_state_or_404(token: str):
    st = STATE.get(token)
    if not st:
        abort(404, "Unknown session token; please re-upload the file.")
    return st


@app.route("/grid/<token>", methods=["GET"])
def grid(token):
    st = _get_state_or_404(token)
    return render_template(
        "grid.html",
        token=token,
        n_ieta=st["n_ieta"],
        n_iphi=st["n_iphi"],
        ieta_min=st["ieta_min"],
        iphi_min=st["iphi_min"],
        matched=len(st["index"]),
        root_path=os.path.basename(st["path"]),
    )


@app.route("/hist", methods=["GET"])
def api_hist():
    token = request.args.get("token")
    st = _get_state_or_404(token)
    try:
        ieta = int(request.args.get("ieta"))
        iphi = int(request.args.get("iphi"))
    except Exception:
        abort(400, "ieta and iphi must be ints")

    key = (ieta, iphi)
    uproot_key = st["index"].get(key)
    if not uproot_key:
        return jsonify({"found": False, "ieta": ieta, "iphi": iphi})

    cache_key = ("hist", uproot_key)
    if cache_key in st["cache"]:
        return jsonify(st["cache"][cache_key])

    try:
        counts, edges = read_histogram(st["path"], uproot_key)
        info = stats_from_hist(counts, edges)
        payload = {
            "found": True,
            "ieta": ieta,
            "iphi": iphi,
            "key": uproot_key,
            "counts": counts.tolist(),
            "edges": edges.tolist(),
            "info": info,
        }
        st["cache"][cache_key] = payload
        return jsonify(payload)
    except Exception as e:
        abort(500, f"Failed to read histogram: {e}")


@app.route("/heatmap", methods=["GET"])
def api_heatmap():
    token  = request.args.get("token")
    metric = (request.args.get("metric") or "mean").lower()
    force  = request.args.get("force") in ("1", "true", "True")

    st = _get_state_or_404(token)
    st.setdefault("heatmap_cache", {})

    # Serve cached unless 'force' is requested
    if not force and metric in st["heatmap_cache"]:
        return jsonify(st["heatmap_cache"][metric])

    grid, vmin, vmax = compute_heatmap(
        st["path"],
        st["index"],
        st["n_ieta"],
        st["n_iphi"],
        st["ieta_min"],
        st["iphi_min"],
        metric,
    )
    payload = {"grid": grid, "vmin": vmin, "vmax": vmax, "metric": metric}
    st["heatmap_cache"][metric] = payload  # refresh cache
    return jsonify(payload)


if __name__ == "__main__":
    app.run(debug=True)

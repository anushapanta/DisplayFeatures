from flask import Flask, render_template, request, redirect, url_for, jsonify, abort
import os, uuid
from cachetools import LRUCache
from src.root_grid import index_histograms, read_histogram, stats_from_hist

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB limit
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Each upload gets a token → we keep its path, index, and a small cache
STATE = {}  # token: {"path": str, "index": dict[(ieta,iphi)->key], "cache": LRUCache}

# Defaults (change if your coordinate base is different)
DEFAULT_N_IETA = 96
DEFAULT_N_IPHI = 256
DEFAULT_IETA_MIN = 0   # set to -48 if your ieta is -48..+47
DEFAULT_IPHI_MIN = 0   # set to 1 if your iphi is 1..256

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# in app.py (replace the upload() body)
@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("rootfile")
    if not f or f.filename == "":
        return redirect(url_for("index"))

    token = str(uuid.uuid4())
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{token}.root")
    f.save(save_path)

    n_ieta   = int(request.form.get("n_ieta") or 96)
    n_iphi   = int(request.form.get("n_iphi") or 256)
    ieta_min = int(request.form.get("ieta_min") or 0)
    iphi_min = int(request.form.get("iphi_min") or 0)
    order    = request.form.get("order") or "ieta-major"

    try:
        mapping = index_histograms(save_path, n_ieta, n_iphi, ieta_min, iphi_min, order)
    except Exception as e:
        return render_template("index.html", error=f"Failed to read ROOT file: {e}")

    STATE[token] = {
        "path": save_path,
        "index": mapping,
        "cache": LRUCache(maxsize=1024),
        "n_ieta": n_ieta,
        "n_iphi": n_iphi,
        "ieta_min": ieta_min,
        "iphi_min": iphi_min,
        "order": order,
    }
    return redirect(url_for("grid", token=token))

    f = request.files.get("rootfile")
    if not f or f.filename == "":
        return redirect(url_for("index"))

    token = str(uuid.uuid4())
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{token}.root")
    f.save(save_path)

    # Build index of (ieta,iphi) -> uproot key
    try:
        mapping = index_histograms(save_path)
    except Exception as e:
        return render_template("index.html", error=f"Failed to read ROOT file: {e}")

    # You can infer grid extents here if needed; for now, use defaults
    STATE[token] = {
        "path": save_path,
        "index": mapping,
        "cache": LRUCache(maxsize=1024),
        "n_ieta": int(request.form.get("n_ieta") or DEFAULT_N_IETA),
        "n_iphi": int(request.form.get("n_iphi") or DEFAULT_N_IPHI),
        "ieta_min": int(request.form.get("ieta_min") or DEFAULT_IETA_MIN),
        "iphi_min": int(request.form.get("iphi_min") or DEFAULT_IPHI_MIN),
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

if __name__ == "__main__":
    app.run(debug=True)

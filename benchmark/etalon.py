"""Real-audio étalon benchmark: engines vs a human-verified score aligned to a real recording.
usage: python benchmark/etalon.py [--engines ...]   -> results/etalon.json, results/etalon.md, results/etalon.png
The copyrighted recording is NOT in the repo: put the separated violin at benchmark/data_real/<set>/audio.wav (see DATASETS.md)."""
import sys, json, csv, time, argparse, re
from pathlib import Path
import numpy as np, mir_eval
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
src = (ROOT / "benchmark/run.py").read_text()
exec(src[src.index("hz = lambda"):src.index("sets = sorted")])   # same metrics as the synthetic benchmark
from archetto_transcribe import engines as E
ap = argparse.ArgumentParser(); ap.add_argument("--engines", default="basicpitch,crepe,pyin,mt3,ours")
a = ap.parse_args(); RES = ROOT / "results"; out = RES / "etalon.json"
res = json.loads(out.read_text()) if out.exists() else {}
for d in sorted((ROOT / "benchmark/data_real").glob("*/")):
    if not (d / "audio.wav").exists(): print("missing audio", d); continue
    ref = [(float(r["onset_s"]), float(r["offset_s"]), int(r["midi"])) for r in csv.DictReader(open(d / "notes.csv"))]
    lo, hi = ref[0][0] - 0.3, max(e for _, e, _ in ref) + 0.3
    crepe = None
    for eng in a.engines.split(","):
        k = f"{d.name}/{eng}"
        if k in res and "error" not in res[k]: continue
        try:
            t0 = time.time()
            est = E.ours(d / "audio.wav", notes=crepe) if (eng == "ours" and crepe) else E.transcribe(d / "audio.wav", eng)
            if eng == "crepe": crepe = est
            est = [x for x in est if lo <= x[0] <= hi]
            r = evaluate(ref, est); r["seconds"] = round(time.time() - t0, 1)
        except Exception as e: r = {"error": f"{type(e).__name__}: {e}"}
        res[k] = r; print(k, r, flush=True); out.write_text(json.dumps(res, indent=1))
keys = ["onset_F1", "onset_offset_F1", "note_F1_no_offset", "pitch_acc", "fragmentation"]
ok = {k: v for k, v in res.items() if "error" not in v}
order = sorted(ok, key=lambda k: -ok[k]["note_F1_no_offset"])
L = ["| rank | set / engine | note onset F1 | onset+offset F1 | note F1 (no offset) | pitch acc. | fragmentation ↓ | est/ref |", "|---|---|---|---|---|---|---|---|"]
for i, k in enumerate(order, 1): L.append(f"| {i} | {k} | " + " | ".join(f"{ok[k][m]:.3f}" for m in keys) + f" | {ok[k]['n_est']}/{ok[k]['n_ref']} |")
(RES / "etalon.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 4.5)); w = 0.2; x = np.arange(len(order))
for j, m in enumerate(["onset_F1", "note_F1_no_offset", "onset_offset_F1"]): ax.bar(x + (j - 1) * w, [ok[k][m] for k in order], w, label=m)
ax.set_xticks(x); ax.set_xticklabels([k.split("/")[1] for k in order]); ax.set_ylim(0, 1); ax.legend(); ax.grid(axis="y", alpha=.3)
ax.set_title("Étalon (real recording): Libertango, violin part, score-aligned ground truth"); fig.tight_layout(); fig.savefig(RES / "etalon.png", dpi=130)

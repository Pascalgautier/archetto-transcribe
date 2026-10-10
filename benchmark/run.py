"""Benchmark: engines x datasets, mir_eval note metrics. usage: python benchmark/run.py [--engines basicpitch,crepe,pyin,mt3,ours]
Writes results/results.json, results/leaderboard.md, results/leaderboard.png."""
import sys, json, csv, time, argparse
from pathlib import Path
import numpy as np, mir_eval
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from archetto_transcribe import engines as E
from archetto_transcribe import pipeline as P

ap = argparse.ArgumentParser(); ap.add_argument("--engines", default="basicpitch,crepe,pyin,mt3,ours"); ap.add_argument("--data", default=str(ROOT / "benchmark/data"))
a = ap.parse_args(); RES = ROOT / "results"; RES.mkdir(exist_ok=True)
res = json.loads((RES / "results.json").read_text()) if (RES / "results.json").exists() else {}
hz = lambda m: 440.0 * 2 ** ((np.asarray(m, float) - 69) / 12)

def evaluate(ref, est):
    ri = np.array([[s, e] for s, e, _ in ref]); rp = hz([p for *_, p in ref])
    if not est: return {"onset_F1": 0, "onset_offset_F1": 0, "note_F1_no_offset": 0, "pitch_acc": 0, "fragmentation": 0, "n_est": 0, "n_ref": len(ref)}
    ei = np.array([[s, max(e, s + 0.01)] for s, e, *_ in est]); ep = hz([x[2] for x in est])
    on = mir_eval.transcription.onset_precision_recall_f1(ri, ei, onset_tolerance=0.05)[2]
    nf = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep, offset_ratio=None)[2]
    nof = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep)[2]
    m = mir_eval.transcription.match_note_onsets(ri, ei, onset_tolerance=0.05)
    pa = float(np.mean([abs(12 * np.log2(rp[i] / ep[j])) < 0.5 for i, j in m])) if m else 0.0
    # fragmentation: estimated notes per matched reference note beyond 1 (extra notes starting inside a reference note)
    extra = sum(max(0, sum(1 for s, e, *_ in est if rs + 0.05 < s < re_ - 0.03)) for rs, re_, _ in ref)
    return {"onset_F1": round(on, 3), "onset_offset_F1": round(nof, 3), "note_F1_no_offset": round(nf, 3), "pitch_acc": round(pa, 3),
            "fragmentation": round(extra / max(1, len(ref)), 3), "n_est": len(est), "n_ref": len(ref)}

sets = sorted(p for p in Path(a.data).glob("*/") if (p / "notes.csv").exists() and (p / "audio.wav").exists())
cache = {}
for d in sets:
    ref = [(float(r["onset_s"]), float(r["offset_s"]), int(r["midi"])) for r in csv.DictReader(open(d / "notes.csv"))]
    for eng in a.engines.split(","):
        k = f"{d.name}/{eng}"
        if k in res and "error" not in res[k]: continue
        try:
            t0 = time.time()
            if eng == "ours" and (d.name, "crepe") in cache: est = E.ours(d / "audio.wav", notes=cache[(d.name, "crepe")])
            else: est = E.transcribe(d / "audio.wav", eng)
            if eng == "crepe": cache[(d.name, "crepe")] = est
            r = evaluate(ref, est); r["seconds"] = round(time.time() - t0, 1)
        except Exception as e:
            r = {"error": f"{type(e).__name__}: {e}"}
        res[k] = r; print(k, r, flush=True); (RES / "results.json").write_text(json.dumps(res, indent=1))

# leaderboard
engs = sorted({k.split("/")[1] for k in res}); keys = ["onset_F1", "onset_offset_F1", "note_F1_no_offset", "pitch_acc", "fragmentation"]
mean = {e: {m: float(np.mean([v[m] for k, v in res.items() if k.endswith("/" + e) and m in v])) for m in keys} for e in engs}
order = sorted(engs, key=lambda e: -mean[e]["onset_offset_F1"])
L = ["| rank | engine | note onset F1 | onset+offset F1 | note F1 (no offset) | pitch acc. | fragmentation ↓ |", "|---|---|---|---|---|---|---|"]
for i, e in enumerate(order, 1):
    L.append(f"| {i} | {e} | " + " | ".join(f"{mean[e][m]:.3f}" for m in keys) + " |")
L += ["", "Per dataset:", "", "| dataset | engine | onset F1 | onset+offset F1 | pitch acc. | fragmentation | est/ref notes | s |", "|---|---|---|---|---|---|---|---|"]
for k, v in sorted(res.items()):
    if "error" in v: L.append(f"| {k.split('/')[0]} | {k.split('/')[1]} | error: {v['error'][:60]} |||||"); continue
    L.append(f"| {k.split('/')[0]} | {k.split('/')[1]} | {v['onset_F1']} | {v['onset_offset_F1']} | {v['pitch_acc']} | {v['fragmentation']} | {v['n_est']}/{v['n_ref']} | {v['seconds']} |")
(RES / "leaderboard.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 4.5)); w = 0.2; x = np.arange(len(order))
for j, m in enumerate(["onset_F1", "onset_offset_F1", "pitch_acc"]):
    ax.bar(x + (j - 1) * w, [mean[e][m] for e in order], w, label=m)
ax.set_xticks(x); ax.set_xticklabels(order); ax.set_ylim(0, 1); ax.set_title("archetto-transcribe benchmark (synthetic violin, mean over datasets)"); ax.legend(); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(RES / "leaderboard.png", dpi=130)

"""Reproducible synthetic violin ground truth: MusicXML part -> (notes.csv, audio.wav).
Additive bowed-string model (harmonics ~1/k with a body-resonance tilt, delayed vibrato, bow attack noise, legato
re-bowing, light room tail). Deterministic (fixed seed). License-clean: the scores are Archetto's own transcriptions."""
import sys, csv
from pathlib import Path
import numpy as np, soundfile as sf, music21 as m21
SR = 22050

def score_notes(xml, part=0, bpm=None):
    s = m21.converter.parse(str(xml)); p = s.parts[part].stripTies()
    mm = s.flatten().getElementsByClass(m21.tempo.MetronomeMark); q = 60.0 / (bpm or (mm[0].number if mm else 120))
    out = []
    for n in p.flatten().notes:
        ps = [n.pitch.midi] if isinstance(n, m21.note.Note) else [x.midi for x in n.pitches][-1:]
        for pm in ps:
            out.append((float(n.offset) * q, float(n.offset + n.quarterLength) * q, int(pm)))
    return sorted(out)

def render(notes, seed=0):
    rng = np.random.default_rng(seed); dur = (max(e for _, e, _ in notes) if notes else 1) + 1.5
    y = np.zeros(int(dur * SR))
    for s, e, p in notes:
        n0, n1 = int(s * SR), int(e * SR); L = max(1, n1 - n0); t = np.arange(L) / SR
        f = 440 * 2 ** ((p - 69) / 12)
        vib = np.where(t > 0.15, 0.22 * np.sin(2 * np.pi * (5.4 + rng.normal(0, .2)) * t) * np.minimum(1, (t - 0.15) / 0.2), 0)
        ph = 2 * np.pi * np.cumsum(f * 2 ** (vib / 12)) / SR
        sig = sum((1 / k) * (1.0 if k < 5 else 0.6) * np.sin(k * ph + rng.uniform(0, 6.28)) for k in range(1, 12) if k * f < SR / 2.2)
        env = np.minimum(1, t / 0.03) * np.minimum(1, (L / SR - t) / 0.03 + 0.02) * (0.85 + 0.15 * np.sin(2 * np.pi * 0.7 * t))
        att = rng.normal(0, 1, L) * np.exp(-t / 0.02) * 0.15
        y[n0:n0 + L] += (sig * env + att) * 0.25 * rng.uniform(0.8, 1.1)
    ir = rng.normal(0, 1, int(0.25 * SR)) * np.exp(-np.arange(int(0.25 * SR)) / (0.06 * SR)) * 0.03
    y = np.convolve(y, np.r_[1, ir])[:len(y)] + rng.normal(0, 1e-3, len(y))
    return y / (np.abs(y).max() + 1e-9) * 0.8

if __name__ == "__main__":
    xml, part, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
    notes = score_notes(xml, part)
    with open(out / "notes.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["onset_s", "offset_s", "midi"]); w.writerows(notes)
    sf.write(out / "audio.wav", render(notes), SR); print(out, len(notes), "notes")

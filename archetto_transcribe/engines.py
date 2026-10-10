"""Pluggable note transcribers. Every engine returns a list of (onset_s, offset_s, midi_pitch, confidence)."""
import numpy as np
from . import pipeline as P

ENGINES = ("basicpitch", "crepe", "pyin", "mt3", "ours")


def _refine(notes, wav, onsets, drop_db=10.0, min_len=0.05, merge_gap=0.03, tol=0.04):
    """Archetto post-processing of monophonic f0 notes:
    1. offsets: a note ends where its energy falls `drop_db` below the note's own peak (bow lifted, rest, reverb tail);
    2. held notes: same pitch, (almost) contiguous and NO audio attack at the junction -> one note (vibrato / bow-change splits);
       a real re-attack (onset within ±tol) keeps repeated notes apart (steady eighths stay eighths);
    3. isolated octave errors (±12 from both neighbours) corrected; blips shorter than min_len removed."""
    import librosa
    y, sr = librosa.load(str(wav), sr=22050, mono=True); hop = 256
    db = 20 * np.log10(np.maximum(librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0], 1e-8))
    tt = librosa.times_like(db, sr=sr, hop_length=hop); ons = np.asarray(onsets, float)
    out = []
    for s, e, p, c in sorted(notes):
        k = (tt >= s) & (tt < e)
        if k.any():
            pk = db[k].max(); i0 = np.argmax(db[k]); idx = np.where(k)[0]
            below = np.where(db[idx[i0]:idx[-1] + 1] < pk - drop_db)[0]
            if len(below): e = max(s + min_len, tt[idx[i0] + below[0]])
        out.append([s, e, p, c])
    merged = []
    for n in out:
        if merged and n[2] == merged[-1][2] and n[0] - merged[-1][1] < merge_gap and not (len(ons) and np.min(np.abs(ons - n[0])) <= tol):
            merged[-1][1] = max(merged[-1][1], n[1]); continue
        merged.append(n)
    for i in range(1, len(merged) - 1):
        a, b, c = merged[i - 1][2], merged[i][2], merged[i + 1][2]
        if abs(b - a) == 12 and abs(b - c) == 12 and a == c: merged[i][2] = a
    return [tuple(n) for n in merged if n[1] - n[0] >= min_len]


def ours(wav, f0_engine="crepe", notes=None):
    """Archetto engine: monophonic f0 (CREPE by default) segmented at audio onsets + Archetto refinement (_refine)."""
    notes = P.transcribe(wav, f0_engine) if notes is None else notes
    return _refine(notes, wav, P.audio_onsets(wav))


def transcribe(wav, engine="ours"):
    engine = engine.lower()
    if engine == "ours":
        return ours(wav)
    return P.transcribe(wav, engine)

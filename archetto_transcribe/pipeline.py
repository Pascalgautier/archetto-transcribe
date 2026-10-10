#!/usr/bin/env python3
"""stem2violin — YouTube -> partie de violon (PDF/MusicXML/MIDI) + backing track sans l'instrument mélodique.

Usage :
  stem2violin <url_youtube | fichier_audio> --start 10:08 [--end 12:16] [--stem auto|guitar|other|vocals|piano|guitar+other]
              [--transpose auto|0|12|-12] [--bars N] [--count-in] [--out DOSSIER] [--title "..."]
"""
import argparse, json, os, re, shutil, subprocess, sys, warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")
import logging; logging.disable(logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]   # package root (profiles/, mt3/)
MSCORE = Path(os.environ.get("MSCORE", "mscore"))
STEMS = ["drums", "bass", "other", "vocals", "guitar", "piano"]
MELODIC_CANDIDATES = ["guitar", "other", "vocals", "piano"]
AUTO_LEAD_CANDIDATES = ["guitar", "other", "piano"]       # vocals: transcribed only if asked explicitly
# 'other' usually mixes a pad/drone and a lead line: only its notes from A4 upward count as "lead"
LEAD_FLOOR = {"guitar": 55, "vocals": 55, "piano": 55, "other": 69}
FRIENDLY_KEYS = {  # (tonic pc, mode): preferred (uses open strings G D A E, or C/F)
    (9, "minor"): True, (4, "minor"): True, (2, "minor"): True, (7, "minor"): True, (11, "minor"): False,
    (0, "major"): True, (7, "major"): True, (5, "major"): True, (2, "major"): True, (9, "major"): True,
    (10, "major"): False, (4, "major"): False}
PC_NAMES = ["C", "C#", "D", "E-", "E", "F", "F#", "G", "A-", "A", "B-", "B"]
VIOLIN_LOW, VIOLIN_HIGH, VIOLIN_COMFORT_HIGH = 55, 100, 93   # G3, E7, A6


def log(*a):
    print("[stem2violin]", *a, file=sys.stderr, flush=True)


import time as _time
TIMINGS = {}
_cur = [None, 0.0]


def mark(stage=None):
    """Stage stopwatch: closes the running stage and opens a new one (report.json -> timings_s)."""
    t = _time.perf_counter()
    if _cur[0]:
        TIMINGS[_cur[0]] = round(TIMINGS.get(_cur[0], 0) + t - _cur[1], 2)
    _cur[0], _cur[1] = stage, t


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"Commande échouée : {' '.join(map(str, cmd))}\n{r.stderr[-2000:]}")
    return r


def parse_time(s):
    if s is None:
        return None
    parts = [float(p) for p in str(s).split(":")]
    t = 0.0
    for p in parts:
        t = t * 60 + p
    return t


def fmt_time(t):
    t = int(round(t))
    h, rem = divmod(t, 3600); m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def start_from_url(url):
    m = re.search(r"[?&]t=(\d+)", url)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------- 1. Source
YTDLP_CLIENTS = [None, "android", "tv", "web_safari", "mweb"]


def yt_meta(url):
    last = ""
    for c in YTDLP_CLIENTS:
        cmd = ["yt-dlp", "--dump-single-json", "--no-warnings", "--no-playlist"]
        if c:
            cmd += ["--extractor-args", f"youtube:player_client={c}"]
        r = subprocess.run(cmd + [url], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            d = json.loads(r.stdout)
            if any(f.get("acodec") not in (None, "none") and f.get("url") for f in d.get("formats", [])):
                return d, c
        last = r.stderr
    raise RuntimeError("yt-dlp n'arrive pas à lire la vidéo :\n" + last[-1500:])


def yt_download(url, client, a, b, dest):
    """Exact-cut download; tries several YouTube clients (some return 403 or 'not a bot' checks)."""
    order = [client] + [c for c in YTDLP_CLIENTS if c != client]
    last = ""
    for c in order:
        cmd = ["yt-dlp", "-f", "bestaudio/best", "--no-warnings", "--no-playlist", "--force-overwrites",
               "--download-sections", f"*{a:.2f}-{b:.2f}", "--force-keyframes-at-cuts", "-o", str(dest)]
        if c:
            cmd += ["--extractor-args", f"youtube:player_client={c}"]
        r = subprocess.run(cmd + [url], capture_output=True, text=True)
        if r.returncode == 0 and Path(dest).exists() and Path(dest).stat().st_size > 10000:
            return
        last = r.stderr
        log(f"  client YouTube {c or 'par défaut'} refusé, essai suivant…")
    raise RuntimeError("Téléchargement impossible :\n" + last[-1500:])


# ---------------------------------------------------------------- 2. Tempo / grid
def beat_grid(stemdir):
    """Steady 4/4 grid from the separated drums; downbeat = beat with the most kick + bass onsets."""
    import librosa, numpy as np
    sr, hop = 22050, 128
    dr, _ = librosa.load(str(stemdir / "drums.wav"), sr=sr)
    ba, _ = librosa.load(str(stemdir / "bass.wav"), sr=sr)
    if np.sqrt(np.mean(dr ** 2)) < 1e-3:   # no drums: fall back to the whole mix
        dr = sum(librosa.load(str(stemdir / f"{s}.wav"), sr=sr)[0] for s in STEMS)
    oe = librosa.onset.onset_strength(y=dr, sr=sr, hop_length=hop)
    times = librosa.frames_to_time(np.arange(len(oe)), sr=sr, hop_length=hop)
    # 1) tempo: autocorrelation peak between 70 and 150 BPM
    ac = librosa.autocorrelate(oe - oe.mean())
    lags = np.arange(int(60 / 150 * sr / hop), int(60 / 70 * sr / hop) + 1)
    L = lags[np.argmax(ac[lags])]
    bpm0 = 60.0 / (L * hop / sr)
    # 2) refine the period with a tight beat tracker + linear fit
    _, beats = librosa.beat.beat_track(onset_envelope=oe, sr=sr, hop_length=hop, units="time", bpm=bpm0, tightness=400)
    i = np.arange(len(beats))
    period, off = np.polyfit(i, beats, 1)
    keep = np.abs(beats - (period * i + off)) < 0.04
    if keep.sum() > 8:
        period, off = np.polyfit(i[keep], beats[keep], 1)
    if abs(60 / period - bpm0) > 5:
        period = 60.0 / bpm0
    dur = times[-1]
    def mean_at(sig, g):
        idx = np.clip(np.searchsorted(times, g), 0, len(sig) - 1)
        return float(np.mean(np.maximum.reduce([sig[np.clip(idx + d, 0, len(sig) - 1)] for d in (-1, 0, 1)])))
    # 3) beat phase: maximise drum onsets on the grid (fine search)
    phases = np.arange(0, period, 0.004)
    ph = phases[int(np.argmax([mean_at(oe, np.arange(p0, dur - 0.05, period)) for p0 in phases]))]
    grid = np.arange(ph, dur - 0.05, period)
    # 4) downbeat: kick + bass onsets
    S = np.abs(librosa.stft(dr, n_fft=1024, hop_length=hop)); f = librosa.fft_frequencies(sr=sr, n_fft=1024)
    kick = np.maximum(0, np.diff(S[f < 150].sum(0), prepend=0))
    Sb = np.abs(librosa.stft(ba, n_fft=1024, hop_length=hop))
    bflux = np.maximum(0, np.diff(Sb.sum(0), prepend=0))
    kick /= kick.max() + 1e-9; bflux /= bflux.max() + 1e-9
    n = min(len(kick), len(bflux))
    sig = kick[:n] + 0.5 * bflux[:n]
    score = [mean_at(sig, grid[k::4]) for k in range(4)]
    k = int(np.argmax(score))
    return float(60.0 / period), float(period), grid, grid[k::4], np.asarray(beats, float)


def beat_map(tracked, db0, period, nbeats, max_dev=0.12):
    """Time (s, from the first downbeat) of every score beat 0..nbeats, following the tracked beats
    (tempo may drift); falls back to the steady grid where tracking is missing or implausible."""
    import numpy as np
    tr = np.sort(np.asarray(tracked, float))
    out = [0.0]
    j = int(np.argmin(np.abs(tr - db0))) if len(tr) else -1
    use = j >= 0 and abs(tr[j] - db0) < 0.25 * period
    for k in range(1, nbeats + 1):
        nxt = out[-1] + period
        if use and len(tr):
            i = int(np.argmin(np.abs(tr - (db0 + nxt))))
            if abs(tr[i] - (db0 + nxt)) < max_dev and 0.7 * period < tr[i] - db0 - out[-1] < 1.4 * period:
                nxt = tr[i] - db0
        out.append(nxt)
    b = np.array(out)
    # smooth beat-to-beat jitter of the tracker (median of 5 inter-beat intervals), keep the long-term drift
    ibi = np.diff(b)
    sm = np.array([np.median(ibi[max(0, i - 2):i + 3]) for i in range(len(ibi))])
    b = np.concatenate([[0.0], np.cumsum(sm)])
    # never drift more than max_dev*4 from the steady grid in total (protects against tracker octave errors)
    lin = np.arange(nbeats + 1) * period
    return np.clip(b, lin - 4 * max_dev, lin + 4 * max_dev)


def trim_stems(stemdir, t0, dur, outdir):
    import soundfile as sf
    outdir.mkdir(parents=True, exist_ok=True)
    for s in STEMS:
        y, sr = sf.read(stemdir / f"{s}.wav")
        a = int(round(t0 * sr)); b = a + int(round(dur * sr))
        sf.write(outdir / f"{s}.wav", y[a:b], sr)
    return outdir


# ---------------------------------------------------------------- 3. Stems
def separate(wav, outdir):
    run([sys.executable, "-m", "demucs", "-n", "htdemucs_6s", "-o", str(outdir), str(wav)])
    return outdir / "htdemucs_6s" / wav.stem


TRANSCRIBER = "basicpitch"     # --transcripteur basicpitch|crepe|pyin|mt3


def transcribe(wav, method=None):
    """Notes (début, fin, midi, amplitude) d'un fichier audio, selon le transcripteur choisi."""
    m = (method or TRANSCRIBER).lower()
    if m == "basicpitch":
        return bp_notes(wav)
    if m == "crepe":
        return crepe_notes(wav)
    if m == "pyin":
        return pyin_notes(wav)
    if m == "mt3":
        return mt3_notes(wav)
    raise ValueError(f"transcripteur inconnu : {m}")


def f0_to_notes(times, f0, conf, wav, min_len=0.06, hold=0.04, onsets=None):
    """Segmentation d'une courbe de f0 monophonique en notes : frontières aux changements de demi-ton stables (>= hold s)
    et aux attaques audio (ré-attaque sur la même note) ; notes de moins de min_len supprimées."""
    import numpy as np, scipy.signal as sg
    times = np.asarray(times, float); f0 = np.asarray(f0, float); conf = np.asarray(conf, float)
    voiced = np.isfinite(f0) & (f0 > 0)
    midi = np.full(len(f0), np.nan); midi[voiced] = 69 + 12 * np.log2(f0[voiced] / 440.0)
    sm = midi.copy()
    if voiced.any():
        fill = np.where(voiced, midi, np.interp(np.arange(len(midi)), np.where(voiced)[0], midi[voiced]))
        sm = np.where(voiced, sg.medfilt(fill, 5), np.nan)
    hop = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
    nh = max(1, int(round(hold / hop)))
    ons = np.asarray(audio_onsets(wav) if onsets is None else onsets, float)
    notes, i, n = [], 0, len(sm)
    while i < n:
        if not np.isfinite(sm[i]):
            i += 1; continue
        j = i; cur = int(round(sm[i]))
        while j + 1 < n and np.isfinite(sm[j + 1]):
            r = int(round(sm[j + 1]))
            if r != cur and np.all(np.round(sm[j + 1:j + 1 + nh][np.isfinite(sm[j + 1:j + 1 + nh])]) == r) and np.isfinite(sm[j + 1:j + 1 + nh]).all():
                break
            j += 1
        t0, t1 = times[i], times[j] + hop
        # ré-attaques audio à l'intérieur de la note (même hauteur répétée)
        cuts = [t for t in ons if t0 + min_len < t < t1 - min_len]
        bounds = [t0] + cuts + [t1]
        for a_, b_ in zip(bounds, bounds[1:]):
            if b_ - a_ >= min_len:
                k = (times >= a_) & (times < b_) & np.isfinite(sm)
                p = int(round(float(np.median(sm[k])))) if k.any() else cur
                notes.append((float(a_), float(b_), p, float(np.mean(conf[k])) if k.any() else 0.5))
        i = j + 1
    return notes


def crepe_notes(wav, model="full"):
    import numpy as np, librosa, torch, torchcrepe
    y, sr = librosa.load(str(wav), sr=16000, mono=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    x = torch.tensor(y)[None]
    # décodeur weighted_argmax : avec viterbi, torchcrepe renvoie une périodicité nulle sur ces pistes (constaté sur Violines de Sangre)
    f0, per = torchcrepe.predict(x, sr, hop_length=160, fmin=50.0, fmax=3000.0, model=model, batch_size=1024,
                                 device="cpu", return_periodicity=True, decoder=torchcrepe.decode.weighted_argmax)
    per = torchcrepe.filter.median(per, 3)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=160)[0][:f0.shape[1]]
    f0 = f0[0].numpy(); per = per[0].numpy()[:len(f0)]
    db = 20 * np.log10(np.maximum(rms, 1e-8)); db = np.pad(db, (0, max(0, len(f0) - len(db))), constant_values=-120)
    f0 = np.where((per > 0.45) & (db > db.max() - 45) & (f0 >= 170), f0, np.nan)   # sous sol3 : basse ou sous-harmonique
    return f0_to_notes(np.arange(len(f0)) * 0.01, f0, per, wav)


def pyin_notes(wav):
    import numpy as np, librosa
    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    f0, vflag, vprob = librosa.pyin(y, fmin=180.0, fmax=3000.0, sr=sr, frame_length=2048, hop_length=256)
    f0 = np.where(vflag, f0, np.nan)
    return f0_to_notes(librosa.times_like(f0, sr=sr, hop_length=256), f0, vprob, wav)


MT3_DIR = Path(os.environ.get("MT3_DIR", str(ROOT / "mt3")))   # venv Python 3.12 + JAX CPU + point de contrôle « mt3 »


def mt3_notes(wav):
    """MT3 (Magenta, transcription multi-instruments) : lancé dans son propre environnement (JAX), notes hors batterie."""
    import tempfile
    py, sc = MT3_DIR / ".v/bin/python", MT3_DIR / "infer_mt3.py"
    if not py.exists() or not sc.exists():
        raise RuntimeError(f"MT3 non installé (attendu dans {MT3_DIR})")
    with tempfile.TemporaryDirectory() as td:
        o = Path(td) / "n.json"
        subprocess.run([str(py), str(sc), str(wav), str(o)], check=True, capture_output=True)
        raw = json.loads(o.read_text())
    return [(float(s_), float(e_), int(p_), (v or 100) / 127) for s_, e_, p_, v, prog, drum in raw if not drum and e_ > s_]


def bp_notes(wav):
    from basic_pitch.inference import predict
    _, _, notes = predict(str(wav), onset_threshold=0.5, frame_threshold=0.3, minimum_note_length=80)
    return [(float(s), float(e), int(p), float(a)) for s, e, p, a, *_ in notes]


def drone_pitches(notes, total_dur):
    """Pitch classes that sound for a large fraction of the time in the low register (tanpura/drone)."""
    import numpy as np
    cover = np.zeros(12)
    for s, e, p, a in notes:
        if p < 60:
            cover[p % 12] += (e - s)
    return {pc for pc in range(12) if cover[pc] > 0.25 * total_dur}


def melodic_score(notes, total_dur):
    drones = drone_pitches(notes, total_dur)
    mel = [n for n in notes if n[2] >= 55 and not (n[2] < 64 and n[2] % 12 in drones)]
    return sum((e - s) * a for s, e, p, a in mel), mel


def choose_stems(stemdir, choice, dur):
    import numpy as np, soundfile as sf
    info = {}
    for s in STEMS:
        y, sr = sf.read(stemdir / f"{s}.wav")
        info[s] = {"rms_db": float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9))}
    notes = {}
    for s in MELODIC_CANDIDATES:
        if info[s]["rms_db"] < -50:
            info[s]["melodic_score"] = 0.0; notes[s] = []; continue
        n = transcribe(stemdir / f"{s}.wav")
        sc, _ = melodic_score(n, dur)
        info[s]["melodic_score"] = float(sc); info[s]["notes"] = len(n); notes[s] = n
    if choice == "auto":
        best = max(MELODIC_CANDIDATES, key=lambda s: info[s]["melodic_score"])
        sel = [best]
    else:
        sel = choice.split("+")
        for s in sel:
            if s not in STEMS:
                raise SystemExit(f"Stem inconnu : {s}")
            if s not in notes:
                notes[s] = transcribe(stemdir / f"{s}.wav")
    return sel, info, notes


def lead_notes(notes, stem):
    return [n for n in notes if n[2] >= LEAD_FLOOR.get(stem, 55)]


def per_bar_lead(notes_by_stem, cands, nbars, barlen, total_dur):
    """For every bar, which stem carries the lead? Primary = best overall; switch only when it is (nearly) silent."""
    import numpy as np
    sc = {}
    for s in cands:
        dr = drone_pitches(notes_by_stem[s], total_dur)
        v = np.zeros(nbars)
        for st, e, p, a in lead_notes(notes_by_stem[s], s):
            if p < 64 and p % 12 in dr:
                continue
            b = int(st // barlen)
            if 0 <= b < nbars:
                v[b] += (e - st) * a
        sc[s] = v
    primary = max(cands, key=lambda s: sc[s].sum())
    act = sc[primary][sc[primary] > 0.3]
    thr = 0.5 * float(np.median(act)) if len(act) else 0.5
    strong = np.where(sc[primary] >= thr)[0]
    first, last = (strong[0], strong[-1]) if len(strong) else (nbars, -1)
    choice = []
    for b in range(nbars):
        if first <= b <= last:
            choice.append(primary)          # inside the primary's section: it keeps the line (gaps become rests)
            continue
        best = max(cands, key=lambda s: sc[s][b])
        choice.append(best if sc[best][b] >= 0.3 else None)
    return choice, primary, {s: [round(float(x), 2) for x in v] for s, v in sc.items()}


def split_lead(wav_in, notes, lead_wav, rest_wav):
    """Harmonic mask: pull the detected lead notes (and their harmonics) out of a stem."""
    import numpy as np, soundfile as sf, librosa
    y, sr = sf.read(wav_in, always_2d=True)
    n_fft, hop = 4096, 512
    lead_ch, rest_ch = [], []
    S0 = librosa.stft(y[:, 0], n_fft=n_fft, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    M = np.zeros(S0.shape, dtype=np.float32)
    nfr = S0.shape[1]
    for st, e, p, a in notes:
        f0 = librosa.midi_to_hz(p)
        t0 = max(0, int((st - 0.04) * sr / hop)); t1 = min(nfr, int((e + 0.25) * sr / hop) + 1)
        h = 1
        while h * f0 < sr * 0.45 and h <= 40:
            fc = h * f0
            lo, hi = fc * 2 ** (-0.45 / 12), fc * 2 ** (0.45 / 12)
            b0 = max(0, np.searchsorted(freqs, lo) - 1); b1 = min(len(freqs), np.searchsorted(freqs, hi) + 1)
            M[b0:b1, t0:t1] = 1.0
            h += 1
    # soften edges in time
    from scipy.ndimage import uniform_filter1d
    M = np.clip(uniform_filter1d(M, size=3, axis=1), 0, 1)
    for c in range(y.shape[1]):
        S = librosa.stft(y[:, c], n_fft=n_fft, hop_length=hop)
        lead_ch.append(librosa.istft(S * M, hop_length=hop, length=len(y)))
        rest_ch.append(librosa.istft(S * (1 - M), hop_length=hop, length=len(y)))
    sf.write(lead_wav, np.stack(lead_ch, 1), sr)
    sf.write(rest_wav, np.stack(rest_ch, 1), sr)


def choose_key(tonic_pc, mode, line, request):
    """Returns (semitone shift, tonic pc, mode). request: auto | original | 'E minor' | 'Em' | 'D' ..."""
    if request in (None, "", "original", "none", "0"):
        return 0, tonic_pc, mode
    if request != "auto":
        m = re.match(r"^\s*([A-Ga-g])([#b-]?)\s*(m|min|minor|mineur|maj|major|majeur)?\s*$", request)
        if not m:
            raise SystemExit(f"--key incompris : {request} (ex. 'E minor', 'Em', 'D', 'auto', 'original')")
        base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[m.group(1).upper()]
        base = (base + (1 if m.group(2) == "#" else -1 if m.group(2) in ("b", "-") else 0)) % 12
        md = "minor" if (m.group(3) or "").startswith("m") and m.group(3) not in ("maj", "major", "majeur") else "major"
        if md != mode:
            log(f"Attention : --key {request} change de mode ({mode} -> {md}) ; seule la transposition est appliquée.")
        sh = ((base - tonic_pc + 6) % 12) - 6
        return sh, base, md
    lo = min((p for _, _, p in line), default=60); hi = max((p for _, _, p in line), default=72)
    best = None
    for (pc, md), pref in FRIENDLY_KEYS.items():
        if md != mode:
            continue
        sh = ((pc - tonic_pc + 6) % 12) - 6
        cost = abs(sh) + (0 if pref else 0.5) - (0.1 if sh > 0 else 0)
        # range: avoid pushing the lowest note under G3 or the top above A6 (octave moves are free for the score)
        span = hi - lo
        if span > VIOLIN_COMFORT_HIGH - VIOLIN_LOW:
            cost += 2
        if best is None or cost < best[0]:
            best = (cost, sh, pc, md)
    return best[1], best[2], best[3]


def pitch_shift(wav_in, wav_out, semitones):
    if semitones == 0:
        shutil.copy(wav_in, wav_out); return
    if not shutil.which("rubberband"):
        raise SystemExit("rubberband introuvable (sudo apt install rubberband-cli)")
    # R3 'fine' engine, tempo untouched (-t 1), no formant preservation (instrumental music)
    run(["rubberband", "-q", "--fine", "-t", "1", "-p", str(semitones), str(wav_in), str(wav_out)])


def mix_wavs(paths, out_wav):
    import numpy as np, soundfile as sf
    acc, sr = None, 44100
    for pth in paths:
        y, sr = sf.read(pth, always_2d=True)
        if acc is None:
            acc = y.copy()
        else:
            n = min(len(acc), len(y)); acc = acc[:n] + y[:n]
    peak = np.max(np.abs(acc))
    if peak > 0.99:
        acc *= 0.99 / peak
    sf.write(out_wav, acc, sr)


def gate_notes(notes, tmpwav, stemdir, sel, period):
    """Drop notes where the melodic stem is (almost) silent: bleed from other instruments, not the melody."""
    import numpy as np, soundfile as sf
    mix_stems(stemdir, sel, tmpwav)
    y, sr = sf.read(tmpwav)
    y = y.mean(1) if y.ndim > 1 else y
    hop = int(period * sr / 2)                       # 8th-note frames
    fr = np.array([np.sqrt(np.mean(y[i:i + hop] ** 2)) + 1e-9 for i in range(0, len(y), hop)])
    db = 20 * np.log10(fr)
    active = np.percentile(db, 90)
    thr = active - 24
    def ok(t0, t1):
        a, b = int(t0 * sr / hop), max(int(t0 * sr / hop) + 1, int(t1 * sr / hop))
        return db[a:b].max() >= thr if a < len(db) else False
    return [n for n in notes if ok(n[0], n[1])]


def mix_stems(stemdir, names, out_wav):
    import numpy as np, soundfile as sf
    acc = None
    for s in names:
        y, sr = sf.read(stemdir / f"{s}.wav")
        acc = y if acc is None else acc + y
    peak = np.max(np.abs(acc))
    if peak > 0.99:
        acc = acc * (0.99 / peak)
    sf.write(out_wav, acc, sr)


def to_mp3(wav, mp3):
    run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3)])


def count_in_mp3(wav, mp3, period, bars=1):
    import numpy as np, soundfile as sf
    y, sr = sf.read(wav)
    n = int(round(period * 4 * bars * sr))
    click = np.zeros((n, y.shape[1] if y.ndim > 1 else 1))
    t = np.arange(int(0.05 * sr)) / sr
    for k in range(4 * bars):
        f = 1600 if k % 4 == 0 else 1100
        c = 0.5 * np.sin(2 * np.pi * f * t) * np.exp(-t * 60)
        i = int(round(k * period * sr))
        click[i:i + len(c)] += c[:, None]
    out = np.concatenate([click, y if y.ndim > 1 else y[:, None]])
    tmp = Path(mp3).with_suffix(".tmp.wav"); sf.write(tmp, out, sr); to_mp3(tmp, mp3); tmp.unlink()


# ---------------------------------------------------------------- 4. Melody -> violin
# Réglages de transcription ajustables par un profil d'apprentissage (stem2violin apprendre … ; --profil / --style)
TUNING = {
    "attaque_flux": 3.0,              # attaque = flux spectral >= x fois la médiane …
    "attaque_niveau": 2.5,            # … et montée de niveau >= x ; sinon le fragment prolonge la note tenue
    "fenetre_double_attaque": 0.2,    # (s) une 2e attaque de même hauteur si tôt après le début = même note
    "penalite_simplicite": 1.0,       # préférence pour les positions simples (temps > croche > double)
    "motif_min": 3,                   # un rythme de mesure doit revenir au moins x fois pour servir de motif
    "motif_tolerance": 0.5,           # (temps) écart toléré entre une attaque et celle du motif
}


def load_profile(path):
    """Profil d'apprentissage (JSON) -> TUNING ; renvoie les suggestions non numériques (grille, octave…)."""
    prof = json.loads(Path(path).read_text())
    params = prof.get("parametres", {})
    for k in TUNING:
        if k in params:
            TUNING[k] = type(TUNING[k])(params[k])
    return prof


def lead_attacks(wav):
    """Attack evidence in the lead audio: SuperFlux-like spectral flux (robust to vibrato) and the level rise."""
    import numpy as np, librosa
    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    hop = 256
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    oe = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S, ref=np.max), sr=sr, hop_length=hop, lag=2, max_size=5)
    rms = librosa.feature.rms(S=S, frame_length=2048, hop_length=hop)[0]
    t = librosa.times_like(oe, sr=sr, hop_length=hop)
    act = rms > 0.1 * rms.max()
    med = float(np.median(oe[act])) if act.any() else 1.0

    def attack(s0):
        w = (t >= s0 - 0.07) & (t <= s0 + 0.07)
        a = (t >= s0 - 0.15) & (t <= s0); b = (t >= s0 - 0.02) & (t <= s0 + 0.12)
        if not w.any() or not a.any() or not b.any():
            return True
        return oe[w].max() / (med + 1e-9) >= TUNING["attaque_flux"] and rms[b].max() / (rms[a].min() + 1e-6) >= TUNING["attaque_niveau"]

    def refine(s0):
        """Basic Pitch onsets tend to be ~1-2 frames early: move to the flux peak (if clear) within -30..+80 ms."""
        w = np.where((t >= s0 - 0.03) & (t <= s0 + 0.08))[0]
        if not len(w):
            return s0
        i = w[np.argmax(oe[w])]
        return float(t[i]) if oe[i] >= 2.5 * med else s0
    attack.refine = refine
    return attack


def clean_lead_notes(notes, lead_wav, drones=frozenset()):
    """Turn Basic Pitch's notes for a (mostly) monophonic bowed/sung lead into one note per real event.
    - simultaneous onsets (double stops, harmonics, octave errors) -> one melody note;
    - fragments of a sustained note (vibrato, bow pressure, portamento) without a real attack -> merged;
    - short weak notes inside a held note (harmonics, sub-octaves, vibrato neighbours) -> dropped."""
    import numpy as np
    attack = lead_attacks(lead_wav)
    ns = sorted((n for n in notes if n[2] >= 48 and not (n[2] < 64 and n[2] % 12 in drones)), key=lambda n: n[0])
    if not ns:
        return []
    med = float(np.median([n[2] for n in ns]))
    # 1) clusters of onsets within 70 ms -> melody note (strong, nearest the line's register, loudest)
    events, cl = [], [ns[0]]
    for n in ns[1:]:
        if n[0] - cl[0][0] <= 0.07:
            cl.append(n)
        else:
            events.append(cl); cl = [n]
    events.append(cl)
    seq = []
    for c in events:
        amax = max(n[3] for n in c)
        strong = [n for n in c if n[3] >= 0.6 * amax]
        s, e, p, a = min(strong, key=lambda n: (abs(n[2] - med) // 7, -n[3]))
        seq.append([attack.refine(min(n[0] for n in c)), e, p, a])
    # 2) merge / drop against the note being held. A held note may be interrupted in Basic Pitch's output by a
    #    weak blip and then resume: its "effective" end follows the chain of same-pitch, attack-less fragments.
    def eff_end(i):
        end, p0 = seq[i][1], seq[i][2]
        for j in range(i + 1, len(seq)):
            s2, e2, p2, _ = seq[j]
            if s2 > end + 0.15:
                break
            if p2 == p0 and not attack(s2):
                end = max(end, e2)
        return end
    out = []
    for i, (s, e, p, a) in enumerate(seq):
        if out:
            h = out[-1]
            h_end = max(h[1], h[4])
            held = s <= h_end + 0.15                        # starts while (or just after) the held note sounds
            dur = e - s
            iv = abs(p - h[2])
            if held and iv == 0 and s - h[0] < TUNING["fenetre_double_attaque"]:          # double onset at the start of a note (bow/accent)
                h[1] = max(h[1], e); h[3] = max(h[3], a); continue
            if held and not attack(s):
                if iv == 0:                                 # same pitch, no attack: same note
                    h[1] = max(h[1], e); h[3] = max(h[3], a); continue
                if iv <= 1 and dur < 0.15:                  # vibrato / bend neighbour
                    h[1] = max(h[1], e); continue
                if (iv % 12 in (0, 7) or iv in (19, 24)) and a < 0.6 * h[3] and h_end > s + 0.5 * dur:
                    continue                                # harmonic or sub-octave of the held note
                if dur < 0.25 and a < 0.6 * h[3] and h_end >= e - 0.05:
                    continue                                # weak blip inside the held note
        out.append([s, e, p, a, eff_end(i)])
    # 3) a held note is cut where the next one starts
    for i in range(len(out) - 1):
        if out[i][1] > out[i + 1][0]:
            out[i][1] = out[i + 1][0]
    return [tuple(n[:4]) for n in out]


def quantize_pos(pos, sub):
    """Nearest grid slot, preferring simple positions (beat > 8th > 16th) when two are almost as close."""
    import math
    pen = [x * TUNING["penalite_simplicite"] for x in {4: (0.0, 0.07, 0.035, 0.07), 2: (0.0, 0.05)}[sub]]
    k0 = math.floor(pos * sub)
    return min((k0, k0 + 1), key=lambda k: abs(pos - k / sub) + pen[k % sub])


def extract_melody(notes, dur, period, sub=4, drones=frozenset(), bmap=None):
    """Monophonic line on a 1/sub-beat grid. Returns list of (slot_start, slot_len, midi).
    bmap = time of each beat (beat tracking); without it, a steady grid of `period`."""
    import numpy as np
    if bmap is None:
        bmap = np.arange(int(round(dur / period)) + 1) * period
    bmap = np.asarray(bmap, float)
    kb = np.arange(len(bmap))
    def beats(t):                                         # seconds -> score beats (piecewise linear)
        if t <= bmap[-1]:
            return float(np.interp(t, bmap, kb))
        return float(kb[-1] + (t - bmap[-1]) / period)
    nslots = (len(bmap) - 1) * sub
    # one candidate per onset slot: loudest note (ties -> the octave nearest the running median)
    slots = {}
    for s, e, p, a in notes:
        if p < 48 or (p < 64 and p % 12 in drones):
            continue
        k = quantize_pos(beats(s), sub)
        if 0 <= k < nslots:
            slots.setdefault(k, []).append((s, e, p, a))
    if not slots:
        return []
    allp = [p for v in slots.values() for _, _, p, _ in v]
    med = float(np.median(allp))
    line = []
    for k in sorted(slots):
        c = slots[k]
        amax = max(n[3] for n in c)
        strong = [n for n in c if n[3] >= 0.6 * amax]
        # sitar/zither harmonics: among strong notes, prefer the one closest to the line's median register
        s, e, p, a = min(strong, key=lambda n: (abs(n[2] - med) // 7, -n[3]))
        line.append([k, max(1, quantize_pos(beats(e), sub) - k), p, a])
    # drop weak isolated blips
    amp_floor = 0.25 * np.percentile([l[3] for l in line], 75)
    line = [l for l in line if l[3] >= amp_floor]
    # durations: until next onset, but not longer than the note itself (+1 slot tolerance)
    out = []
    for i, (k, ln, p, a) in enumerate(line):
        nxt = line[i + 1][0] if i + 1 < len(line) else nslots
        d = min(nxt - k, ln + 1)
        if nxt - k - d <= 1:   # fill tiny gaps, avoids 16th rests everywhere
            d = nxt - k
        out.append((k, max(1, d), p))
    # merge zither/sitar tremolo: >=3 rapid repeats of the same pitch -> one long note
    merged = []
    for k, d, p in out:
        if merged and merged[-1][2] == p and merged[-1][0] + merged[-1][1] == k and d == 1 and merged[-1][1] <= 2 * sub:
            merged[-1] = (merged[-1][0], merged[-1][1] + d, p)
        else:
            merged.append((k, d, p))
    return merged



# ---------------------------------------------------------------- 4c. Tango : figures rythmiques de l'accompagnement
# en double-croches (4 par temps), relatives au temps d'arrivée (0) ; « rests » = positions qui doivent être muettes
TANGO_FIGURES = [
    ("demi-soupir + 3 doubles", (-2, -1, 0), (-4, -3)),       # 𝄾 ♬ -> temps : demi-soupir, deux doubles, arrivée sur le temps
    ("4 doubles en levée", (-4, -3, -2, -1, 0), ()),           # ♬♬ -> temps
    ("contretemps", (2,), (0,)),                               # 𝄾 ♪ : croche sur la 2e moitié du temps, rien sur le temps
]


def audio_onsets(wav):
    import numpy as np, librosa
    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    hop = 256
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    oe = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S, ref=np.max), sr=sr, hop_length=hop, lag=2, max_size=5)
    fr = librosa.onset.onset_detect(onset_envelope=oe, sr=sr, hop_length=hop, units="frames", delta=0.07, wait=3)
    return librosa.frames_to_time(fr, sr=sr, hop_length=hop)


def tango_rhythm(line, sub, raw, onsets, bmap, period, tol=0.1, log_=None, name="Violon"):
    """A priori rythmique du tango : là où les attaques mesurées dans l'audio (onset audio confirmé par une note Basic Pitch
    de CETTE voix, à ±60 ms) tombent toutes à ±tol temps des positions d'une figure (demi-soupir + 3 doubles, 4 doubles en
    levée, contretemps) et qu'aucune attaque ne tombe sur ses silences, la fenêtre est réécrite selon la figure (hauteurs
    prises dans les notes Basic Pitch). Renvoie (ligne, rapport)."""
    import numpy as np
    if sub != 4 or not line:
        return line, {"appliquees": [], "partielles": 0, "fenetres": 0}
    bmap = np.asarray(bmap, float); kb = np.arange(len(bmap))
    beats = lambda t: float(np.interp(t, bmap, kb)) if t <= bmap[-1] else float(kb[-1] + (t - bmap[-1]) / period)
    starts = sorted((n[0], n[2], n[3]) for n in raw)
    st_t = np.array([x[0] for x in starts]) if starts else np.zeros(0)
    conf = []
    for t in onsets:
        if len(st_t):
            j = int(np.argmin(np.abs(st_t - t)))
            if abs(st_t[j] - t) <= 0.06:
                conf.append((beats(t), starts[j][1]))
    ob = np.array([c[0] for c in conf]) if conf else np.zeros(0)
    nb = len(bmap) - 1
    L = {k: [d, p] for k, d, p in line}
    used_slots = set(); applied = []; partial = 0; tested = 0
    for fig, ons, rests in TANGO_FIGURES:
        for B in range(1, nb):
            k0 = 4 * B
            if any((k0 + o) in used_slots for o in ons):
                continue
            hit = []
            for o in ons:
                tgt = B + o / 4
                j = np.where(np.abs(ob - tgt) <= tol)[0] if len(ob) else []
                hit.append(int(j[np.argmin(np.abs(ob[j] - tgt))]) if len(j) else None)
            nh = sum(h is not None for h in hit)
            if nh < len(ons):
                if nh >= max(2, len(ons) - 1):
                    partial += 1
                continue
            if any(len(ob) and np.any(np.abs(ob - (B + r / 4)) <= tol * 0.8) for r in rests):
                continue
            # attaques en trop à l'intérieur de la figure (autre rythme) -> non
            lo, hi = B + min(ons + rests) / 4 - tol, B + max(ons) / 4 + tol
            inside = set(np.where((ob >= lo) & (ob <= hi))[0])
            if inside - set(hit):
                continue
            tested += 1
            slots = [k0 + o for o in ons]
            cur = sorted(k for k in L if slots[0] + (min(rests) if rests else 0) - (0 if rests else 0) <= k <= slots[-1])
            before = {k: tuple(L[k]) for k in cur}
            # silences de la figure : couper la note qui déborde dedans
            span0 = k0 + min(list(ons) + list(rests))
            for k in sorted(L):
                if k < span0 < k + L[k][0]:
                    L[k][0] = span0 - k
            for k in cur:
                if span0 <= k <= slots[-1] and k not in slots:
                    del L[k]
            for i_, k in enumerate(slots):
                p = conf[hit[i_]][1]
                if fig == "contretemps":
                    d = 2
                elif i_ + 1 < len(slots):
                    d = slots[i_ + 1] - k
                else:                                   # note d'arrivée : garde sa durée si elle existait
                    d = L[k][0] if k in L else 2
                    nxt = [kk for kk in L if kk > k]
                    if nxt:
                        d = min(d, min(nxt) - k)
                if k in L and fig != "contretemps" and i_ + 1 == len(slots):
                    L[k][0] = max(1, d)                 # hauteur de la transcription gardée pour l'arrivée
                else:
                    L[k] = [max(1, d), p]
            after = {k: tuple(L[k]) for k in sorted(L) if span0 <= k <= slots[-1]}
            if after != {k: v for k, v in before.items() if span0 <= k}:
                applied.append((B // 4 + 1, fig))
            used_slots.update(slots)
    out = sorted((k, d, p) for k, (d, p) in L.items())
    # pas de chevauchement
    fixed = []
    for i_, (k, d, p) in enumerate(out):
        nxt = out[i_ + 1][0] if i_ + 1 < len(out) else None
        fixed.append((k, max(1, min(d, nxt - k)) if nxt is not None else d, p))
    rep = {"appliquees": applied, "partielles": partial, "fenetres": tested}
    if log_:
        from collections import Counter
        c = Counter(f for _, f in applied)
        log_(f"   rythme tango — {name} : {len(applied)} figure(s) écrite(s) " + (f"({', '.join(f'{v}× {k}' for k, v in c.items())}) " if c else "") +
             f"aux mesures {sorted({b for b, _ in applied})} ; {partial} fenêtre(s) presque conformes laissées telles quelles")
    return fixed, rep

def voice_role_rhythm(line, raw, onsets, bmap, period, bars, role, tol=0.12, log_=None, name="Violon", force=()):
    """Rôle rythmique imposé à une voix d'accompagnement (a priori de rôle, demandé pour le tango) :
    role="croches" -> croches régulières (attaques sur chaque temps et chaque contretemps) ;
    role="contretemps" -> demi-soupir + croche sur chaque temps. Dans les mesures `bars` où la voix joue, la ligne est
    réécrite selon ce rythme ; hauteurs = note Basic Pitch de cette voix qui sonne à cet instant (sinon la note transcrite
    qui couvre la position, sinon la précédente). Rapport : pour chaque mesure, part des attaques imposées confirmées
    par une attaque audio (±tol temps) — c'est là que le rythme « colle » à l'enregistrement."""
    import numpy as np
    bmap = np.asarray(bmap, float); kb = np.arange(len(bmap))
    beats = lambda t: float(np.interp(t, bmap, kb)) if t <= bmap[-1] else float(kb[-1] + (t - bmap[-1]) / period)
    tim = lambda b: float(np.interp(b, kb, bmap)) if b <= kb[-1] else float(bmap[-1] + (b - kb[-1]) * period)
    ob = np.array([beats(t) for t in onsets]) if len(onsets) else np.zeros(0)
    L = {k: [d, p] for k, d, p in line}
    offs = (0, 2) if role == "croches" else (2,)
    rep = {}; prev = None
    for bar in sorted(bars):
        k0, k1 = 16 * (bar - 1), 16 * bar
        cur = {k: v for k, v in L.items() if k0 <= k < k1}
        if not cur and bar not in force and not any(k < k0 < k + v[0] for k, v in L.items()):
            continue                                            # (force : passages à 3 violons, la voix joue toujours)                                            # la voix se tait dans cette mesure : on n'invente rien
        cover = lambda q: next((v[1] for k, v in sorted(L.items()) if k <= q < k + v[0]), None)
        new = {}
        for B in range(4):
            for o in offs:
                q = k0 + 4 * B + o; t = tim(q / 4)
                act = [n for n in raw if n[0] - 0.06 <= t < n[1]]
                nxt = next((v[1] for k, v in sorted(L.items()) if k >= q), None)
                p = max(act, key=lambda n: n[0])[2] if act else (cover(q) or prev or nxt)
                if p is None:
                    continue
                new[q] = [2, int(p)]; prev = int(p)
        for k in list(L):
            if k0 <= k < k1: del L[k]
            elif k < k0 < k + L[k][0]: L[k][0] = k0 - k     # la note précédente s'arrête à la barre
        L.update(new)
        hits = [q for q in new if len(ob) and np.min(np.abs(ob - q / 4)) <= tol]
        rests = [k0 + 4 * B for B in range(4)] if role == "contretemps" else []
        on_rest = sum(1 for q in rests if len(ob) and np.min(np.abs(ob - q / 4)) <= tol * 0.8)
        rep[bar] = {"attaques": len(new), "confirmees": len(hits), "attaques_sur_silences": on_rest}
    out = sorted((k, d, p) for k, (d, p) in L.items())
    out = [(k, max(1, min(d, out[i + 1][0] - k)) if i + 1 < len(out) else d, p) for i, (k, d, p) in enumerate(out)]
    good = [b for b, r in rep.items() if r["attaques"] and r["confirmees"] / r["attaques"] >= 0.6]
    summ = {"role": role, "mesures": sorted(rep), "colle_aux_attaques": good,
            "taux_global": round(sum(r["confirmees"] for r in rep.values()) / max(1, sum(r["attaques"] for r in rep.values())), 2),
            "detail": {str(b): r for b, r in rep.items()}}
    if log_:
        log_(f"   rôle rythmique — {name} = {role} : {len(rep)} mesure(s) réécrite(s) ; attaques confirmées par l'audio "
             f"{int(100 * summ['taux_global'])} % ; colle (≥60 %) aux mesures {good}")
    return out, summ

# ---------------------------------------------------------------- 4b. Two violins (duo) in the lead stem
HARM_IV = (12, 19, 24, 28, 31)


def duo_bars(lead_wav, bmap, nbars, min_bars=2):
    """Bars where the lead stem holds two independent lines (not one violin + its harmonics).
    Per half-beat: CQT peaks that are not harmonics of a stronger lower peak; a bar is 'duo' when at least half of its
    half-beats have two such pitches. A duo = >= min_bars consecutive bars with >= 5 such half-beats, extended to neighbouring
    bars with >= 4, at least 3 bars in all, and both lines must move (a drone or sitar sympathetic strings do not count;
    a chord on one entry is not a duo)."""
    import numpy as np, librosa
    y, sr = librosa.load(str(lead_wav), sr=22050, mono=True)
    hop = 256
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop, fmin=librosa.note_to_hz("C3"), n_bins=60))
    tt = librosa.times_like(C[0], sr=sr, hop_length=hop); m0 = 48
    gmax = float(np.percentile(C.max(0), 95))
    kb = np.arange(len(bmap))
    poly = np.zeros((nbars, 8), bool); poly3 = np.zeros((nbars, 8), bool)
    pairs = {}                                         # (bar, half-beat) -> (lower, upper) midi
    for bar in range(nbars):
        for h in range(8):
            a = np.interp(4 * bar + h / 2, kb, bmap); b = np.interp(4 * bar + h / 2 + 0.5, kb, bmap)
            w = (tt >= a) & (tt < b)
            if not w.any():
                continue
            v = np.median(C[:, w], axis=1)
            if v.max() < 0.1 * gmax:
                continue
            pk = [i for i in range(1, 59) if v[i] >= v[i - 1] and v[i] >= v[i + 1] and v[i] >= 0.25 * v.max()]
            keep = []
            for i in sorted(pk):
                if any((i - j) in HARM_IV and v[j] >= 0.5 * v[i] for j in keep):
                    continue
                keep.append(i)
            keep = [i for i in keep if 55 <= i + m0 <= 90]
            poly[bar, h] = len(keep) >= 2
            poly3[bar, h] = len(keep) >= 3
            if len(keep) >= 2:
                top = sorted(sorted(keep, key=lambda i: -v[i])[:2])
                pairs[(bar, h)] = (top[0] + m0, top[1] + m0)
    cnt = poly.sum(1)
    core = cnt >= 5
    runs, b = [], 0
    while b < nbars:
        if core[b]:
            e = b
            while e + 1 < nbars and core[e + 1]:
                e += 1
            if e - b + 1 >= min_bars:
                while b > 0 and cnt[b - 1] >= 4:            # extend to neighbouring bars that are half duo
                    b -= 1
                while e + 1 < nbars and cnt[e + 1] >= 4:
                    e += 1
                # two *independent* lines: each moves (>= 2 pitches) -- a drone or sympathetic strings do not
                lo = {pairs[k][0] for k in pairs if b <= k[0] <= e}
                hi = {pairs[k][1] for k in pairs if b <= k[0] <= e}
                if len(lo) >= 2 and len(hi) >= 2 and e - b + 1 >= 3:
                    runs.append((b + 1, e + 1))
            b = e + 1
        else:
            b += 1
    # trois violons : >= 4 demi-temps à trois hauteurs indépendantes, sur >= 2 mesures, étendu aux voisines (>= 3), dans un duo
    c3 = poly3.sum(1); in_duo = np.zeros(nbars, bool)
    for x, y_ in runs:
        in_duo[x - 1:y_] = True
    trio, b = [], 0
    while b < nbars:
        if c3[b] >= 4 and in_duo[b]:
            e = b
            while e + 1 < nbars and c3[e + 1] >= 3 and in_duo[e + 1]:
                e += 1
            while b > 0 and c3[b - 1] >= 3 and in_duo[b - 1]:
                b -= 1
            if e - b + 1 >= 2:
                trio.append((b + 1, e + 1))
            b = e + 1
        else:
            b += 1
    merged = []
    for x, y_ in trio:                                 # trous d'une mesure comblés
        if merged and x - merged[-1][1] <= 2:
            merged[-1] = (merged[-1][0], y_)
        else:
            merged.append((x, y_))
    return runs, poly, merged


def assign_voices_n(notes, nv_at, runs_t=(), prev_pitch=None):
    """N violons (2 ou 3) : répartition par continuité de hauteur. Notes simultanées -> voix différentes (affectation de coût
    minimal : distance à la dernière hauteur de chaque voix, voix encore en train de sonner pénalisée, croisements pénalisés).
    Puis, dans chaque passage polyphonique (runs_t = [(t0, t1)]), les voix sont renumérotées par importance mélodique
    (amplitude x durée, + continuité avec la mélodie qui précède) : la mélodie principale = Violon I."""
    import itertools
    ns = sorted(notes, key=lambda n: (n[0], n[2]))
    ok = [n for n in ns if not any(m is not n and m[0] < n[1] - 0.03 and m[1] > n[0] + 0.03 and (n[2] - m[2]) in HARM_IV and m[3] >= n[3] for m in ns)]
    V = 3
    vs = [[] for _ in range(V)]; last = [None] * V; end = [-1.0] * V; reg = [None] * V
    i = 0
    while i < len(ok):
        cl = [ok[i]]
        while i + len(cl) < len(ok) and ok[i + len(cl)][0] - ok[i][0] <= 0.07:
            cl.append(ok[i + len(cl)])
        i += len(cl)
        s0 = cl[0][0]; nv = max(1, min(V, nv_at(s0)))
        byp = {}
        for n in cl:                                     # une note par hauteur (la plus forte)
            if n[2] not in byp or n[3] > byp[n[2]][3]:
                byp[n[2]] = n
        cand = sorted(byp.values(), key=lambda n: -n[3])[:nv]
        best, bm = None, None
        for perm in itertools.permutations(range(nv), len(cand)):
            c = 0.0
            for n, v in zip(cand, perm):
                c += abs(n[2] - last[v]) if last[v] is not None else 5.0
                if end[v] > n[0] + 0.06 and last[v] != n[2]:
                    c += 7.0
                c += 0.3 * v
            for (n1, v1), (n2, v2) in itertools.combinations(zip(cand, perm), 2):
                if reg[v1] is not None and reg[v2] is not None and (n1[2] - n2[2]) * (reg[v1] - reg[v2]) < 0:
                    c += 3.0                             # croisement de voix
            if best is None or c < best:
                best, bm = c, perm
        for n, v in zip(cand, bm):
            vs[v].append(n); last[v] = n[2]; end[v] = n[1]
            reg[v] = n[2] if reg[v] is None else 0.8 * reg[v] + 0.2 * n[2]
    # renumérotation par passage : Violon I = mélodie principale
    out = [[] for _ in range(V)]
    bounds = sorted(runs_t) or [(-1e9, 1e9)]
    def seg_of(t):
        for k, (a, b) in enumerate(bounds):
            if a - 0.2 <= t < b:
                return k
        return None
    for k, (a, b) in enumerate(bounds):
        sal = []
        for v in range(V):
            nn = [n for n in vs[v] if seg_of(n[0]) == k]
            sc = sum(n[3] * min(1.5, n[1] - n[0]) for n in nn)
            if nn and prev_pitch is not None:
                first = min(nn, key=lambda n: n[0])
                if first[0] - a < 1.5:
                    sc *= 1.0 + 0.5 * max(0.0, 1 - abs(first[2] - prev_pitch) / 7)
            sal.append(sc)
        order = sorted(range(V), key=lambda v: -sal[v])
        for rank, v in enumerate(order):
            out[rank] += [n for n in vs[v] if seg_of(n[0]) == k]
    out[0] += [n for v in range(V) for n in vs[v] if seg_of(n[0]) is None]
    return [sorted(x) for x in out]


def assign_voices(notes):
    """Split polyphonic Basic Pitch notes into two lines by voice-leading: when two notes sound together the upper one
    is voice 2 (the added violin, above the tune) and the lower one voice 1; a note alone goes to the line whose last
    pitch is nearest. Harmonics of a simultaneous stronger lower note are dropped."""
    ns = sorted(notes, key=lambda n: (n[0], n[2]))
    ok = []
    for n in ns:
        s, e, p, a = n
        if any(m is not n and m[0] < e - 0.03 and m[1] > s + 0.03 and (p - m[2]) in HARM_IV and m[3] >= a for m in ns):
            continue
        ok.append(n)
    v = {1: [], 2: []}
    last = {1: None, 2: None}
    i = 0
    while i < len(ok):
        cl = [ok[i]]
        while i + len(cl) < len(ok) and ok[i + len(cl)][0] - ok[i][0] <= 0.07:
            cl.append(ok[i + len(cl)])
        i += len(cl)
        pitches = sorted({n[2] for n in cl})
        for n in cl:
            s, e, p, a = n
            if len(pitches) >= 2:
                k = 2 if p == pitches[-1] else 1 if p == pitches[0] else None
            else:
                sound = {vv: [m for m in v[vv] if m[1] > s + 0.06] for vv in (1, 2)}
                recent = {vv: [m for m in v[vv] if m[1] > s - 0.3 and m[2] == p] for vv in (1, 2)}
                if recent[2] and not recent[1] and not (sound[2] and not sound[1] and p != sound[2][-1][2]):
                    k = 2                                   # same pitch as the 2nd violin's last note: it goes on
                elif recent[1] and not recent[2]:
                    k = 1
                elif sound[1] and not sound[2] and p < min(m[2] for m in sound[1]) and \
                        all(s - m[0] < 0.25 for m in sound[1]):
                    for m in sound[1]:                      # near-simultaneous entry below: the earlier note was violin 2
                        v[1].remove(m); v[2].append(m); last[2] = m[2]
                    k = 1
                elif sound[1] and not sound[2]:
                    k = 2 if p > max(m[2] for m in sound[1]) else 1
                elif sound[2] and not sound[1]:
                    k = 1 if p < min(m[2] for m in sound[2]) else 2
                elif e - s < 0.3 and last[1] is not None:
                    k = 1                                   # short note alone: an ornament of the tune (voice 1)
                else:
                    d = {vv: abs(p - last[vv]) if last[vv] is not None else 99 for vv in (1, 2)}
                    k = 1 if d[1] <= d[2] else 2
            if k:
                v[k].append(n); last[k] = p
    return v[1], v[2]


def split_voices_audio_n(wav_in, voices, t_from, t_to, outs):
    """N voix : masques harmoniques guidés par la partition (comme split_voices_audio) ; voix 1 = le reste."""
    import numpy as np, soundfile as sf, librosa
    from scipy.ndimage import uniform_filter1d
    y, sr = sf.read(wav_in, always_2d=True)
    n_fft, hop = 4096, 512
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    S = [librosa.stft(y[:, c], n_fft=n_fft, hop_length=hop) for c in range(y.shape[1])]
    mag = np.mean([np.abs(x) for x in S], axis=0)
    ft = librosa.frames_to_time(np.arange(mag.shape[1]), sr=sr, hop_length=hop, n_fft=n_fft)
    df = freqs[1]
    T = []
    for notes in voices:
        Tv = np.zeros(mag.shape, np.float32)
        for s, e, p, a in notes:
            f0n = librosa.midi_to_hz(p)
            for j in np.where((ft >= s - 0.03) & (ft <= e + 0.05))[0]:
                best, bf = -1, f0n
                for c in np.arange(-0.6, 0.61, 0.05):
                    f0 = f0n * 2 ** (c / 12); hs = np.arange(1, 9) * f0
                    sc = np.interp(hs[hs < sr / 2], freqs, mag[:, j]).sum()
                    if sc > best:
                        best, bf = sc, f0
                h = 1
                while h * bf < min(sr * 0.45, 12000):
                    fc = h * bf; sig = max(1.6 * df, fc * (2 ** (0.12 / 12) - 1))
                    lo, hi = np.searchsorted(freqs, fc - 3 * sig), np.searchsorted(freqs, fc + 3 * sig)
                    Tv[lo:hi, j] += np.exp(-0.5 * ((freqs[lo:hi] - fc) / sig) ** 2) / np.sqrt(h)
                    h += 1
        T.append(Tv)
    span = (ft >= t_from) & (ft < t_to)
    den = sum(t_ ** 2 for t_ in T) + 0.05 ** 2
    rest = y.copy(); q = {}
    harm = []
    for v in range(1, len(voices)):
        M = T[v] ** 2 / den; M[:, ~span] = 0.0
        M = np.clip(uniform_filter1d(M, size=3, axis=1), 0, 1)
        o = np.stack([librosa.istft(x * M, hop_length=hop, n_fft=n_fft, length=len(y)) for x in S], 1)
        sf.write(outs[v], o, sr); rest = rest - o; harm.append(o)
    sf.write(outs[0], rest, sr)
    tracks = [rest] + harm
    # fuite : énergie des harmoniques propres de la voix v dans la piste w, relative à sa propre piste
    Z = [np.mean([np.abs(librosa.stft(t_[:, c], n_fft=n_fft, hop_length=hop)) for c in range(t_.shape[1])], axis=0) ** 2 for t_ in tracks]
    def own(v, w):
        tot = 0.0
        for s, e, p, a in voices[v]:
            js = np.where((ft >= s) & (ft <= e) & span)[0]
            others = [q_ for u in range(len(voices)) if u != v for q_ in voices[u] if q_[0] < e and q_[1] > s]
            for h in range(1, 9):
                fc = h * librosa.midi_to_hz(p)
                if any(abs(fc / (hh * librosa.midi_to_hz(o_[2])) - 1) < 0.03 for o_ in others for hh in range(1, 12)):
                    continue
                b = np.searchsorted(freqs, fc); tot += float(Z[w][max(0, b - 2):b + 3][:, js].sum())
        return tot
    R_ = ["I", "II", "III"]
    for v in range(len(voices)):
        for w in range(len(voices)):
            if v != w and voices[v]:
                q[f"violon{R_[v]}_dans_piste_violon{R_[w]}_dB"] = round(10 * np.log10((own(v, w) + 1e-12) / (own(v, v) + 1e-12)), 1)
    return q


def split_voices_audio(wav_in, v1, v2, t_from, t_to, out1, out2):
    """Score-informed harmonic masking: inside [t_from, t_to] each STFT bin goes to the voice whose (vibrato-tracked)
    harmonic comb explains it; bins shared by both are split by a 1/h amplitude prior; what neither explains (bow noise,
    reverb) stays with voice 1. Outside the span voice 1 = the whole stem. out1 + out2 == input."""
    import numpy as np, soundfile as sf, librosa
    y, sr = sf.read(wav_in, always_2d=True)
    n_fft, hop = 4096, 512
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    S = [librosa.stft(y[:, c], n_fft=n_fft, hop_length=hop) for c in range(y.shape[1])]
    mag = np.mean([np.abs(x) for x in S], axis=0)
    nfr = mag.shape[1]
    ft = librosa.frames_to_time(np.arange(nfr), sr=sr, hop_length=hop, n_fft=n_fft)
    T = {1: np.zeros(mag.shape, np.float32), 2: np.zeros(mag.shape, np.float32)}
    df = freqs[1]
    for vv, notes in ((1, v1), (2, v2)):
        for s, e, p, a in notes:
            f0n = librosa.midi_to_hz(p)
            for j in np.where((ft >= s - 0.03) & (ft <= e + 0.05))[0]:
                # vibrato: best comb within +-0.6 semitone
                best, bf = -1, f0n
                for c in np.arange(-0.6, 0.61, 0.05):
                    f0 = f0n * 2 ** (c / 12)
                    hs = np.arange(1, 9) * f0
                    sc = np.interp(hs[hs < sr / 2], freqs, mag[:, j]).sum()
                    if sc > best:
                        best, bf = sc, f0
                h = 1
                while h * bf < min(sr * 0.45, 12000):
                    fc = h * bf
                    sig = max(1.6 * df, fc * (2 ** (0.12 / 12) - 1))
                    lo, hi = np.searchsorted(freqs, fc - 3 * sig), np.searchsorted(freqs, fc + 3 * sig)
                    T[vv][lo:hi, j] += np.exp(-0.5 * ((freqs[lo:hi] - fc) / sig) ** 2) / np.sqrt(h)
                    h += 1
    span = (ft >= t_from) & (ft < t_to)
    floor = 0.05
    M2 = (T[2] ** 2) / (T[1] ** 2 + T[2] ** 2 + floor ** 2)
    M2[:, ~span] = 0.0
    from scipy.ndimage import uniform_filter1d
    M2 = np.clip(uniform_filter1d(M2, size=3, axis=1), 0, 1)
    o2 = np.stack([librosa.istft(x * M2, hop_length=hop, n_fft=n_fft, length=len(y)) for x in S], 1)
    sf.write(out2, o2, sr)
    sf.write(out1, y - o2, sr)
    # quality figures: energy at each voice's own (non-shared) harmonics, in each output
    def comb_energy(sig_out, notes, other):
        Z = np.mean([np.abs(librosa.stft(sig_out[:, c], n_fft=n_fft, hop_length=hop)) for c in range(sig_out.shape[1])], axis=0) ** 2
        tot = 0.0
        for s, e, p, a in notes:
            js = np.where((ft >= s) & (ft <= e) & span)[0]
            for h in range(1, 9):
                fc = h * librosa.midi_to_hz(p)
                if any(abs(fc / (hh * librosa.midi_to_hz(q[2])) - 1) < 0.03 for q in other if q[0] < e and q[1] > s for hh in range(1, 12)):
                    continue                               # shared with the other voice: not informative
                b = np.searchsorted(freqs, fc)
                tot += float(Z[max(0, b - 2):b + 3][:, js].sum())
        return tot
    yy = y.copy()
    q = {}
    for name, sig_out in (("v1", y - o2), ("v2", o2)):
        e1, e2 = comb_energy(sig_out, v1, v2), comb_energy(sig_out, v2, v1)
        q[name] = (e1, e2)
    db = lambda a_, b_: round(10 * np.log10((a_ + 1e-12) / (b_ + 1e-12)), 1)
    return {"violon2_dans_piste_violon1_dB": db(q["v1"][1], q["v2"][1]),   # V2 harmonics: in V1 track vs in V2 track
            "violon1_dans_piste_violon2_dB": db(q["v2"][0], q["v1"][0])}   # V1 harmonics: in V2 track vs in V1 track


def motif_pass(line, sub, nbars, motifs=None, min_count=None, log_=None):
    """Recurring rhythmic motif -> consistent notation.
    1) motif = bar rhythm (onsets/durations inside the bar) that occurs exactly in >= min_count bars;
    2) a bar whose notes, grouped by pitch (repeated notes and rests inside a held note merged), give the same number
       of notes as the motif with every onset within half a beat of the motif's -> rewritten with the motif's rhythm,
       keeping its pitches. A long note anticipated by a 16th before the barline counts as starting on the downbeat.
    3) the short notes that end the motif (e.g. two 8ths after a dotted half) are slurred.
    Returns (line, slurs [(slot_from, slot_to)], motifs, changed bars)."""
    from collections import Counter
    BL = 4 * sub
    tol = max(1, round(TUNING["motif_tolerance"] * sub))   # half a beat by default
    min_count = min_count or int(TUNING["motif_min"])
    notes = sorted([list(n) for n in line])

    def bars_of(ns):
        by = {}
        for n in ns:
            by.setdefault(n[0] // BL, []).append(n)
        return by

    def rhythm(bn, b):
        out = []
        for k, d, p in bn:
            o = k - b * BL
            out.append((o, min(d, BL - o)))
        return tuple(out)

    by = bars_of(notes)
    if motifs is None:
        cnt = Counter(rhythm(v, b) for b, v in by.items() if len(v) >= 2)
        motifs = [list(r) for r, c in cnt.most_common(2) if c >= min_count]
    if not motifs:
        return [tuple(n) for n in notes], [], [], []
    # anticipations: a long note starting in the last 16th before a barline is read as starting on the barline
    antic = {}
    for i, n in enumerate(notes):
        k, d, p = n
        nb = (k // BL + 1) * BL
        if 0 < nb - k <= max(1, sub // 4) and k + d >= nb + sub:
            antic[i] = nb
    out_notes = [list(n) for n in notes]
    slurs, changed = [], []
    for b in range(nbars):
        start, end = b * BL, (b + 1) * BL
        cand = []
        for i, n in enumerate(out_notes):
            k = antic.get(i, n[0])
            if start <= k < end and not (i in antic and antic[i] != k):
                cand.append((i, k))
            elif i in antic and antic[i] == start:
                cand.append((i, start))
        # notes starting in this bar, minus those anticipating the next bar
        cand = [(i, k) for i, k in cand if not (i in antic and antic[i] == end)]
        cand.sort(key=lambda x: x[1])
        if len(cand) < 2:
            continue
        groups = []                      # [pitch, onset, [note indices]]
        for i, k in cand:
            p = out_notes[i][2]
            if groups and groups[-1][0] == p:
                groups[-1][2].append(i)
            else:
                groups.append([p, k - start, [i]])
        last_i = cand[-1][0]
        last_end = (antic.get(last_i, out_notes[last_i][0])) + out_notes[last_i][1] - (antic.get(last_i, out_notes[last_i][0]) - out_notes[last_i][0])
        best = None
        for m in motifs:
            if len(m) != len(groups):
                continue
            err = [abs(g[1] - o) for g, (o, _) in zip(groups, m)]
            if max(err) <= tol and (end - min(last_end, end)) <= sub:
                if best is None or sum(err) < best[0]:
                    best = (sum(err), m)
        if best is None:
            continue
        m = best[1]
        if [(g[1], ) for g in groups] == [(o, ) for o, _ in m] and \
                all(len(g[2]) == 1 for g in groups) and not any(i in antic for i, _ in cand):
            pass                         # already in motif rhythm (durations may still differ)
        overflow = max(0, out_notes[last_i][0] + out_notes[last_i][1] - end)
        for g, (o, dm) in zip(groups, m):
            i0 = g[2][0]
            out_notes[i0][0] = start + o
            out_notes[i0][1] = dm
            for j in g[2][1:]:
                out_notes[j] = None
        # the last motif note keeps a tie into the next bar if the original did
        out_notes[groups[-1][2][0]][1] += overflow
        # previous note (in the previous bar) runs up to the downbeat if its successor was anticipated
        first_i = groups[0][2][0]
        if first_i in antic:
            prev = [n for n in out_notes if n is not None and n[0] < start and n[0] + n[1] >= start - max(1, sub // 4) and n is not out_notes[first_i]]
            for n in prev:
                n[1] = start - n[0]
        out_notes = [n for n in out_notes if n is not None] if False else out_notes
        # slur the trailing short notes (<= 1 beat) that follow a long one (>= 2 beats)
        tail = []
        for g, (o, dm) in reversed(list(zip(groups, m))):
            if dm <= sub:
                tail.insert(0, start + o)
            else:
                break
        if len(tail) >= 2 and len(tail) < len(m):
            slurs.append((tail[0], tail[-1]))
        changed.append(b + 1)
        # rebuild the anticipation index (positions changed)
        out_notes_c = [n for n in out_notes if n is not None]
        idx_map = {id(n): j for j, n in enumerate(out_notes_c)}
        antic = {idx_map[id(out_notes[i])]: v for i, v in antic.items() if out_notes[i] is not None and out_notes[i][0] != v}
        out_notes = out_notes_c
    out_notes.sort()
    # a long note anticipated just before the barline that ends a rewritten bar starts on the downbeat
    for n in out_notes:
        nb = (n[0] // BL + 1) * BL
        if 0 < nb - n[0] <= max(1, sub // 4) and n[0] + n[1] >= nb + sub and (nb // BL) in changed:
            n[1] -= nb - n[0]; n[0] = nb
    out_notes.sort()
    for i, n in enumerate(out_notes[:-1]):        # the motif's last note then runs to the barline again
        nxt = out_notes[i + 1]
        if (n[0] // BL + 1) in changed and nxt[0] % BL == 0 and nxt[0] - (n[0] + n[1]) == max(1, sub // 4) and n[0] // BL == (nxt[0] - 1) // BL:
            n[1] = nxt[0] - n[0]
    # no overlaps
    for a, b_ in zip(out_notes, out_notes[1:]):
        if a[0] + a[1] > b_[0]:
            a[1] = b_[0] - a[0]
    return [tuple(n) for n in out_notes if n[1] > 0], slurs, motifs, changed


def smooth_octaves(line):
    """Fold isolated octave jumps (harmonics / octave errors of the transcription) back into the line."""
    out = [list(x) for x in line]
    for _ in range(2):
        for i in range(len(out)):
            prev = out[i - 1][2] if i > 0 else None
            nxt = out[i + 1][2] if i + 1 < len(out) else None
            ctx = [x for x in (prev, nxt) if x is not None]
            if not ctx:
                continue
            p = out[i][2]
            ref = sum(ctx) / len(ctx)
            if abs(p - ref) >= 9:
                cand = min((p - 12, p, p + 12), key=lambda q: abs(q - ref))
                out[i][2] = cand
    return [tuple(x) for x in out]


def violin_transpose(line, mode):
    import numpy as np
    ps = np.array([p for _, _, p in line])
    if mode != "auto":
        shift = int(mode)
    else:
        # by octaves only: put the median of the line in the violin's singing register (D4..E5 ideally)
        med = float(np.median(ps)) if len(ps) else 70
        shift = 0
        while med + shift < 62: shift += 12
        while med + shift > 81: shift -= 12
    res = []
    for k, d, p in line:
        q = p + shift
        while q < VIOLIN_LOW: q += 12
        while q > VIOLIN_HIGH: q -= 12
        res.append((k, d, q))
    return res, shift


def detect_key(stemdir, sel, line):
    """Tonic = strongest pitch class of the bass (or drone); mode from the melody's 3rd above the tonic."""
    import numpy as np, librosa
    y, sr = librosa.load(str(stemdir / "bass.wav"), sr=22050, mono=True)
    if np.sqrt(np.mean(y ** 2)) < 1e-3:
        y = sum(librosa.load(str(stemdir / f"{s}.wav"), sr=22050, mono=True)[0] for s in STEMS if s != "drums")
    ch = librosa.feature.chroma_cqt(y=y, sr=sr, fmin=librosa.note_to_hz("C1"), n_octaves=5).sum(1)
    w = np.zeros(12)
    for k, d, p in line:
        w[p % 12] += d
    # bass histogram as main evidence, melody as tie-breaker
    score = ch / (ch.max() + 1e-9) + 0.3 * w / (w.max() + 1e-9)
    tonic = int(np.argmax(score))
    allw = w + 0.5 * ch / (ch.max() + 1e-9) * (w.max() + 1e-9)
    # mode : profils de Krumhansl-Kessler à tonique fixée (tierce, sixte et sensible comptent toutes) —
    # la seule tierce trompe quand la mélodie l'évite (ex. tango en ré mineur : si♭, do♯, jamais de fa)
    maj = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    mnr = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    h = np.roll(allw, -tonic)
    mode = "minor" if np.corrcoef(h, mnr)[0, 1] >= np.corrcoef(h, maj)[0, 1] else "major"
    return tonic, mode


KK_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KK_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def key_plan(stemdir, used, line, sub, nbars, bmap, main_pc, main_mode, min_bars=8, change_cost=2.5):
    """Modulations : tonalité de chaque mesure (harmonie de l'accompagnement + mélodie, fenêtre de 4 mesures,
    profils de Krumhansl-Kessler), lissée par Viterbi (coût d'un changement) ; sections d'au moins min_bars mesures.
    Renvoie [(mesure_0based, pc, mode), ...] — une seule entrée s'il n'y a pas de modulation."""
    import numpy as np, librosa
    harm = [s_ for s_ in STEMS if s_ != "drums" and s_ not in used and (stemdir / f"{s_}.wav").exists()]
    y = None
    for s_ in harm:
        a_, sr = librosa.load(str(stemdir / f"{s_}.wav"), sr=22050, mono=True)
        y = a_ if y is None else y[:len(a_)] + a_[:len(y)]
    hop = 2048
    ch = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop, fmin=librosa.note_to_hz("C1"), n_octaves=6)
    t = librosa.times_like(ch, sr=sr, hop_length=hop)
    B = np.zeros((nbars, 12)); M = np.zeros((nbars, 12))
    for b in range(nbars):
        w = (t >= bmap[4 * b]) & (t < bmap[min(4 * b + 4, len(bmap) - 1)])
        if w.any():
            B[b] = ch[:, w].sum(1)
    for k, d, p_ in line:
        b = k // (4 * sub)
        if b < nbars:
            M[b, p_ % 12] += d
    E = B / (B.max(1, keepdims=True) + 1e-9) + 0.5 * M / (M.max(1, keepdims=True) + 1e-9)
    W = np.array([E[max(0, b - 1):b + 3].sum(0) for b in range(nbars)])
    keys = [(pc, md) for md in ("minor", "major") for pc in range(12)]
    prof = {"major": np.array(KK_MAJ), "minor": np.array(KK_MIN)}
    S = np.zeros((nbars, 24))
    for b in range(nbars):
        if W[b].sum() < 1e-6:
            continue
        for j, (pc, md) in enumerate(keys):
            S[b, j] = np.corrcoef(np.roll(W[b], -pc), prof[md])[0, 1] + (0.05 if (pc, md) == (main_pc, main_mode) else 0) \
                      + (0.03 if md == main_mode else 0)
    # Viterbi
    cost = np.zeros(24); back = np.zeros((nbars, 24), int)
    cost = -S[0].copy()
    for b in range(1, nbars):
        stay = cost; move = cost.min() + change_cost
        back[b] = np.where(stay <= move, np.arange(24), int(cost.argmin()))
        cost = np.minimum(stay, move) - S[b]
    path = [int(cost.argmin())]
    for b in range(nbars - 1, 0, -1):
        path.append(int(back[b][path[-1]]))
    path = path[::-1]
    secs = []
    for b, j in enumerate(path):
        if not secs or secs[-1][1] != j:
            secs.append([b, j])
    # sections trop courtes -> fondues dans la voisine
    changed = True
    while changed and len(secs) > 1:
        changed = False
        for i, (b0, j) in enumerate(secs):
            b1 = secs[i + 1][0] if i + 1 < len(secs) else nbars
            if b1 - b0 < min_bars:
                if i == 0:
                    secs[1][0] = 0
                secs.pop(i); changed = True
                break
        secs = [x for k_, x in enumerate(secs) if k_ == 0 or x[1] != secs[k_ - 1][1]]
    return [(b, keys[j][0], keys[j][1]) for b, j in secs]


def key_from_pc(pc, mode):
    from music21 import key
    names_flat = ["C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B"]
    names_sharp = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    opts = []
    for nm in {names_flat[pc], names_sharp[pc]}:
        k = key.Key(nm if mode == "major" else nm.lower(), mode)
        opts.append((abs(k.sharps), k.sharps > 0, k))  # fewer accidentals; tie -> flats
    return sorted(opts, key=lambda x: (x[0], x[1]))[0][2]


def tidy_rests(part):
    """Merge consecutive rests, then split them on beat boundaries (readable 'orchestral part' rests)."""
    from music21 import stream, note
    for m in part.getElementsByClass(stream.Measure):
        els = list(m.notesAndRests)
        # merge adjacent rests
        i = 0
        while i < len(els) - 1:
            a, b = els[i], els[i + 1]
            if a.isRest and b.isRest:
                a.quarterLength += b.quarterLength; m.remove(b); els.pop(i + 1)
            else:
                i += 1
        for r in [e for e in m.notesAndRests if e.isRest]:
            off, ql = r.offset, r.quarterLength
            if ql == 4.0 or (ql == 2.0 and off in (0.0, 2.0)) or (ql == 3.0 and off in (0.0, 1.0) and False):
                continue
            pieces, t = [], off
            end = off + ql
            while t < end - 1e-6:
                frac = t - int(t + 1e-6)
                if abs(frac - 0.25) < 1e-6 or abs(frac - 0.75) < 1e-6:
                    nb = min(end, t + 0.25)
                elif abs(frac - 0.5) < 1e-6:
                    nb = min(end, t + 0.5)
                else:
                    nb = min(end, (int(t + 1e-6) + 1) if frac > 1e-6 else t + 1)
                # whole half-bar chunk if possible
                if abs(t - round(t)) < 1e-6 and t in (0.0, 2.0) and end - t >= 2.0:
                    nb = t + 2.0
                pieces.append((t, nb - t)); t = nb
            if len(pieces) > 1:
                m.remove(r)
                for o, d in pieces:
                    nr = note.Rest(); nr.quarterLength = d; m.insert(o, nr)


NOTE_VALUES = (4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25)


def split_value(off, dur):
    """Split a held note starting at `off` (quarters, 4/4) into the simplest readable values:
    long values only from the strong beats, syncopations limited to the usual 8th-quarter-8th / 16th-8th-16th."""
    out = []
    while dur > 1e-6:
        fo = round(off % 1, 3)
        for v in NOTE_VALUES:
            if v > dur + 1e-9 or off + v > 4 + 1e-9:
                continue
            end = round((off + v) % 1, 3)
            if fo == 0:
                ok = v <= 1.5 or (v == 2.0 and off in (0, 1, 2)) or (v == 3.0 and off == 0) or (v == 4.0 and off == 0)
            elif fo == 0.5:
                ok = v in (0.25, 0.5) or (v == 1.0 and end == 0.5) or (v == 1.5 and end == 0)
            else:
                ok = v == 0.25 or (fo == 0.25 and (v == 0.5 or v == 0.75))
            if ok:
                break
        else:
            v = 0.25
        out.append(v); off += v; dur -= v
    return out


def simplify_ties(part):
    """Re-split chains of tied notes inside each bar (music21 gives e.g. half~quarter~16th for 3¼ beats;
    a violinist reads dotted half~16th)."""
    from music21 import stream, note, tie
    for m in part.getElementsByClass(stream.Measure):
        ns = [n for n in m.notesAndRests]
        i = 0
        while i < len(ns):
            n = ns[i]
            if not n.isNote:
                i += 1; continue
            j = i
            while (j + 1 < len(ns) and ns[j + 1].isNote and ns[j].tie is not None and ns[j].tie.type in ("start", "continue")
                   and ns[j + 1].pitch.midi == n.pitch.midi and abs(ns[j].offset + ns[j].quarterLength - ns[j + 1].offset) < 1e-6):
                j += 1
            chain = ns[i:j + 1]
            off, tot = float(chain[0].offset), float(sum(c.quarterLength for c in chain))
            vals = split_value(off, tot)
            if [float(c.quarterLength) for c in chain] != vals:
                incoming = chain[0].tie is not None and chain[0].tie.type in ("stop", "continue")
                outgoing = chain[-1].tie is not None and chain[-1].tie.type in ("start", "continue")
                for c in chain:
                    m.remove(c)
                o = off
                for k, v in enumerate(vals):
                    nn = note.Note(chain[0].pitch); nn.quarterLength = v
                    prev, nxt = (k > 0 or incoming), (k < len(vals) - 1 or outgoing)
                    nn.tie = tie.Tie("continue" if prev and nxt else "stop" if prev else "start") if (prev or nxt) else None
                    m.insert(o, nn); o += v
            i = j + 1


def build_score(line, sub, bpm, nbars, title, subtitle, out_base, tonic_pc=None, mode="minor", response_text=None,
                line2=None, slurs=(), slurs2=(), kplan=None, line3=None, slurs3=()):
    """MusicXML + MIDI. With line2: two staves, Violon I and Violon II (Violon II rests until it enters)."""
    import copy
    from music21 import stream, metadata, key, note, tempo
    if tonic_pc is None:
        tmp = stream.Part()
        for k, d, m in line:
            n = note.Note(m); n.quarterLength = d / sub; tmp.insert(k / sub, n)
        k_an = tmp.flatten().notes.stream().analyze("key") if line else key.Key("C")
        tonic_pc, mode = k_an.tonic.pitchClass, k_an.mode
    ks = key_from_pc(tonic_pc, mode)
    # plan tonal : [(mesure 0-based, Key)] ; la 1re entrée est l'armure de départ
    kp = [(b, key_from_pc(pc, md)) for b, pc, md in kplan] if kplan and len(kplan) > 1 else None
    if kp:
        ks = kp[0][1]
    duo = bool(line2)
    parts = [_make_part(line, sub, bpm, nbars, ks, "Violon I" if duo else "Violon", "Vl. I" if duo else "Vl.", True, response_text, slurs, kp)]
    if duo:
        parts.append(_make_part(line2, sub, bpm, nbars, ks, "Violon II", "Vl. II", False, None, slurs2, kp))
    if duo and line3:
        parts.append(_make_part(line3, sub, bpm, nbars, ks, "Violon III", "Vl. III", False, None, slurs3, kp))
    sc = stream.Score()
    sc.metadata = metadata.Metadata()
    sc.metadata.title = title
    sc.metadata.composer = subtitle
    for p in parts:
        sc.insert(0, p)
    sc.write("musicxml", fp=str(out_base) + ".musicxml")
    sc.write("midi", fp=str(out_base) + ".mid")
    if duo:      # separate parts for each player
        for p, nm in zip(parts, ("violon_I", "violon_II", "violon_III")):
            s1 = stream.Score(); s1.metadata = metadata.Metadata()
            s1.metadata.title = title.replace("Partie de violon", {"violon_I": "Violon I", "violon_II": "Violon II", "violon_III": "Violon III"}[nm])
            s1.metadata.composer = subtitle
            pc = copy.deepcopy(p)
            m1 = pc.getElementsByClass(stream.Measure).first()
            if m1 is not None and not m1.getElementsByClass(tempo.MetronomeMark):
                m1.insert(0, tempo.MetronomeMark(number=round(bpm), referent=1.0))
            s1.insert(0, pc)
            s1.write("musicxml", fp=str(Path(out_base).parent / nm) + ".musicxml")
    return ks


def _spell(m, ks):
    """Orthographe des notes d'une mesure dans la tonalité ks : notes de la gamme avec leur nom ; notes étrangères,
    orthographe fonctionnelle : en mineur 2de mineure (mi♭ en ré m.), 3ce majeure (fa♯), 4te augmentée (sol♯),
    6te et 7e montées (si♮, do♯) ; en majeur 2de/3ce/6te/7e mineures en bémols, 4te augmentée en dièse."""
    from music21 import interval as _iv
    scale_names = {pp.pitchClass: pp.name for pp in ks.getScale(ks.mode).getPitches()}
    _chrom = ["m2", "M3", "A4", "M6", "M7"] if ks.mode == "minor" else ["m2", "m3", "A4", "m6", "m7"]
    chrom_names = {}
    for nm_ in _chrom:
        p_ = ks.tonic.transpose(_iv.Interval(nm_))
        if p_.pitchClass not in scale_names:
            chrom_names[p_.pitchClass] = p_.name
    for n in m.recurse().notes:
        pc = n.pitch.pitchClass
        if pc in scale_names and n.pitch.name != scale_names[pc]:
            n.pitch = n.pitch.getEnharmonic() if n.pitch.getEnharmonic().name == scale_names[pc] else n.pitch
        elif pc in chrom_names and n.pitch.name != chrom_names[pc]:
            alt = [e for e in [n.pitch] + n.pitch.getAllCommonEnharmonics() if e.name == chrom_names[pc]]
            if alt:
                n.pitch = alt[0]
    for n in m.recurse().notes:
        if n.pitch.accidental is not None and n.pitch.accidental.alter == 0:
            n.pitch.accidental = None
        if n.pitch.accidental is not None:
            n.pitch.accidental.displayStatus = None
            n.pitch.accidental.displayType = "normal"


def _response_marks(p, response_text):
    from music21 import stream, expressions
    mlist = list(p.getElementsByClass(stream.Measure))
    # mesure « vide » = aucune attaque de note (une note liée qui déborde de la phrase ne compte pas)
    empty = [not any(n_.tie is None or n_.tie.type == "start" for n_ in m_.notes) for m_ in mlist]
    i = 1
    while i < len(mlist):
        if empty[i] and not empty[i - 1]:
            j = i
            while j < len(mlist) and empty[j]:
                j += 1
            if 2 <= j - i <= 4 and j < len(mlist):     # réponse entre deux phrases (pas l'intro ni la fin)
                te = expressions.TextExpression(response_text); te.placement = "above"; te.style.fontStyle = "italic"
                mlist[i].insert(0, te)
            i = j
        else:
            i += 1


def _make_part(line, sub, bpm, nbars, ks, name, abbr, with_tempo, response_text, slurs=(), kp=None):
    from music21 import stream, note, meter, tempo, instrument, clef, expressions
    p = stream.Part()
    vi = instrument.Violin(); vi.instrumentName = name; vi.instrumentAbbreviation = abbr
    p.insert(0, vi)
    p.partName = name; p.partAbbreviation = abbr
    p.insert(0, clef.TrebleClef())
    p.insert(0, meter.TimeSignature("4/4"))
    if with_tempo:
        p.insert(0, tempo.MetronomeMark(number=round(bpm), referent=1.0))
    q = 1.0 / sub
    total_ql = nbars * 4.0
    for k, d, m in line:
        off = k * q
        if off >= total_ql:
            break
        dur = min(d * q, total_ql - off)
        n = note.Note(m); n.quarterLength = dur
        p.insert(off, n)
    p.insert(0, ks.__class__(ks.tonic.name, ks.mode))
    slur_q = [(k1 / sub, k2 / sub) for k1, k2 in slurs]
    p.makeRests(fillGaps=True, inPlace=True, timeRangeFromBarDuration=True)
    if p.highestTime < total_ql:
        r = note.Rest(); r.quarterLength = total_ql - p.highestTime; p.append(r)
    p = p.makeMeasures()
    p.makeTies(inPlace=True)
    tidy_rests(p)
    simplify_ties(p)
    measures_ = list(p.getElementsByClass(stream.Measure))
    key_at = [ks] * len(measures_)
    if kp:                                   # changements d'armure aux modulations
        for i_, (b0, k_) in enumerate(kp):
            b1 = kp[i_ + 1][0] if i_ + 1 < len(kp) else len(measures_)
            for b_ in range(b0, min(b1, len(measures_))):
                key_at[b_] = k_
            if 0 < b0 < len(measures_):
                from music21 import key as _key
                old = measures_[b0].getElementsByClass(_key.KeySignature)
                for o_ in old:
                    measures_[b0].remove(o_)
                measures_[b0].insert(0, _key.KeySignature(k_.sharps))
    for m_, k_ in zip(measures_, key_at):
        _spell(m_, k_)
    for m_, k_ in zip(measures_, key_at):
        m_.makeAccidentals(useKeySignature=k_, overrideStatus=True, cautionaryNotImmediateRepeat=False,
                           cautionaryPitchClass=False, inPlace=True)
        m_.makeBeams(inPlace=True)
    if response_text:
        _response_marks(p, response_text)
    if slur_q:
        from music21 import spanner
        heads = {}
        for n in p.recurse().notes:
            if n.tie is None or n.tie.type == "start":
                heads.setdefault(round(float(n.getOffsetInHierarchy(p)), 4), n)
        for a_, b_ in slur_q:
            n1, n2 = heads.get(round(a_, 4)), heads.get(round(b_, 4))
            if n1 is not None and n2 is not None and n1 is not n2:
                p.insert(0, spanner.Slur(n1, n2))
    return p


STYLE = ROOT / "styles" / "partie_orchestre.mss"


def render(musicxml, pdf, png, hide_empty=False, preview=True):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    st = ["-S", str(STYLE)] if STYLE.exists() else []
    if hide_empty and STYLE.exists():
        # duo score: a staff that only rests in a system is hidden (Violon II appears where it plays)
        import tempfile
        sty = Path(tempfile.gettempdir()) / f"stem2violin_duo_{os.getpid()}.mss"
        sty.write_text(STYLE.read_text().replace("<Style>", "<Style>\n    <hideEmptyStaves>1</hideEmptyStaves>\n"
                                                 "    <dontHideStavesInFirstSystem>1</dontHideStavesInFirstSystem>", 1))
        st = ["-S", str(sty)]
    for old in Path(png).parent.glob(Path(png).stem + "-*.png"):
        old.unlink()
    mscz = Path(musicxml).with_suffix(".mscz")
    run([str(MSCORE), *st, "-o", str(mscz), str(musicxml)], env=env)
    run([str(MSCORE), "-o", str(pdf), str(mscz)], env=env)
    if not preview:
        return
    run([str(MSCORE), "-o", str(png), "-T", "20", "-r", "110", str(mscz)], env=env)
    if shutil.which("pdftoppm"):
        run(["pdftoppm", "-r", "150", "-png", "-f", "1", "-l", "1", "-singlefile", str(pdf),
             str(Path(pdf).parent / "apercu_page1")])


def export_pdf(argv):
    """PDF (MuseScore) d'une partition MusicXML, par ex. celle téléchargée depuis l'éditeur du lecteur."""
    ap = argparse.ArgumentParser(prog="stem2violin pdf", description="Exporter une partition MusicXML en PDF avec MuseScore.")
    ap.add_argument("musicxml", help="fichier .musicxml (par ex. *_modifiee.musicxml téléchargé depuis le lecteur)")
    ap.add_argument("-o", "--out", help="PDF de sortie (par défaut : même nom, extension .pdf)")
    ap.add_argument("--apercu", action="store_true", help="écrire aussi un aperçu PNG de la première page")
    a = ap.parse_args(argv)
    src = Path(a.musicxml).resolve()
    if not src.exists():
        sys.exit(f"Fichier introuvable : {src}")
    pdf = Path(a.out).resolve() if a.out else src.with_suffix(".pdf")
    duo = src.read_text(errors="ignore").count("<score-part ") > 1      # duo : la portée vide est masquée comme dans violin_part.pdf
    render(src, pdf, pdf.with_suffix(".png"), hide_empty=duo, preview=False)
    if a.apercu and shutil.which("pdftoppm"):
        run(["pdftoppm", "-r", "150", "-png", "-f", "1", "-l", "1", "-singlefile", str(pdf), str(pdf.with_name(pdf.stem + "_apercu"))])
    log("PDF :", pdf)
    return 0


AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac"}


def import_stems(src_dir, stem_map, t0, t1, outdir):
    """Pistes séparées par un autre outil -> les 6 fichiers attendus (noms Demucs), coupés comme le segment.
    Plusieurs fichiers vers la même piste sont additionnés ; une piste absente devient du silence."""
    import numpy as np, soundfile as sf
    mp = dict(x.split("=", 1) for x in stem_map.split(",") if "=" in x)
    mp = {k.strip().lower(): v.strip() for k, v in mp.items()}
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    acc, n = {}, 0
    files = sorted(f for f in Path(src_dir).iterdir() if f.suffix.lower() in AUDIO_EXT)
    if not files:
        raise SystemExit(f"aucun fichier audio dans {src_dir}")
    for f in files:
        name = f.stem.lower()
        target = mp.get(name) or next((v for k, v in mp.items() if k in name), None) \
            or (name if name in STEMS else next((s_ for s_ in STEMS if s_ in name), "other"))
        if target not in STEMS:
            raise SystemExit(f"--stem-map : piste inconnue « {target} » (attendu : {', '.join(STEMS)})")
        tmp = outdir / f"_import_{name}.wav"
        run(["ffmpeg", "-v", "error", "-y", "-i", str(f), "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}",
             "-vn", "-ac", "2", "-ar", "44100", str(tmp)])
        y, sr = sf.read(tmp, dtype="float32"); tmp.unlink()
        n = max(n, len(y))
        if y.ndim == 1:
            y = np.stack([y, y], axis=1)
        if target in acc:
            a_ = acc[target]; m_ = max(len(a_), len(y))
            pad_ = lambda z: np.vstack([z, np.zeros((m_ - len(z), 2), dtype="float32")])
            y = pad_(a_) + pad_(y)
        acc[target] = y
        log(f"   {f.name} -> {target}")
    for s_ in STEMS:
        y = acc.get(s_, np.zeros((n, 2), dtype="float32"))
        if len(y) < n:
            y = np.vstack([y, np.zeros((n - len(y), 2), dtype="float32")])
        sf.write(outdir / f"{s_}.wav", y, 44100)
    return outdir


def digital_silences(wav, t0, min_len=5.0):
    """Plages de silence numérique (échantillons ~nuls) d'au moins min_len s, en temps absolu de la source."""
    import numpy as np, soundfile as sf
    y, sr = sf.read(str(wav), dtype="float32")
    y = np.abs(y).max(axis=1) if y.ndim > 1 else np.abs(y)
    n = len(y) // sr
    quiet = [bool(y[i * sr:(i + 1) * sr].max() < 1e-4) for i in range(n)]
    spans, i = [], 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if j - i >= min_len:
                spans.append((round(t0 + i, 1), round(t0 + j, 1)))
            i = j
        else:
            i += 1
    return spans


# ---------------------------------------------------------------- main
def main():
    import numpy as np
    if len(sys.argv) > 1 and sys.argv[1] == "site":       # stem2violin site add|build|remove ...
        raise SystemExit("'site' belongs to the Archetto web app, not to archetto-transcribe")
    if len(sys.argv) > 1 and sys.argv[1] == "apprendre":  # stem2violin apprendre original.musicxml corrige.musicxml
        import apprendre
        return apprendre.cli(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "couper":     # stem2violin couper dossier --mesures 1-8,120-140
        raise SystemExit("'couper' (razor) belongs to the Archetto web app, not to archetto-transcribe")
    if len(sys.argv) > 1 and sys.argv[1] == "pdf":        # stem2violin pdf partition.musicxml [-o sortie.pdf]
        return export_pdf(sys.argv[2:])
    ap = argparse.ArgumentParser(description="Partie de violon + backing track à partir d'une vidéo YouTube.")
    ap.add_argument("source", help="URL YouTube ou fichier audio/vidéo local")
    ap.add_argument("--start", help="début (mm:ss ou secondes) ; par défaut le ?t= de l'URL, sinon 0")
    ap.add_argument("--end", help="fin (mm:ss) ; par défaut début + 2:00")
    ap.add_argument("--stems-dir", help="pistes déjà séparées (ex. stems ElevenLabs) : dossier de fichiers audio, utilisé à la place de Demucs")
    ap.add_argument("--stem-map", default="", help="correspondance nom_de_fichier=piste, ex. 'sitar=guitar,percussion=drums' ; "
                    "pistes reconnues : drums bass other vocals guitar piano ; un fichier inconnu va dans 'other'")
    ap.add_argument("--downbeat", type=float, help="instant (s, depuis --start) de la 1re mesure, si on le connaît (ex. 0 pour un morceau généré)")
    ap.add_argument("--response-text", help="texte écrit sur les mesures de réponse (2 à 4 mesures vides après une phrase), "
                    "ex. 'Violon : répétez / improvisez'")
    ap.add_argument("--stem", default="auto", help="auto | guitar | other | vocals | piano | combinaison ex. guitar+other")
    ap.add_argument("--transpose", default="auto", help="auto (octaves, tessiture violon) ou demi-tons, ex. 12")
    ap.add_argument("--grid", type=int, default=None, choices=[2, 4], help="quantification : 4 = double-croches (défaut), 2 = croches")
    ap.add_argument("--style", help="style du morceau (tango, sitar…) : charge profils/<style>.json appris avec « stem2violin apprendre »")
    ap.add_argument("--profil", help="profil d'apprentissage JSON à utiliser (prioritaire sur --style)")
    ap.add_argument("--count-in", action="store_true", help="ajoute une version du backing track avec 1 mesure de clics")
    ap.add_argument("--key", default="auto", help="tonalité cible : auto (tonalité « violon » la plus proche), original, ou ex. 'E minor', 'Em', 'D'")
    ap.add_argument("--backing", default="clean", choices=["clean", "simple"],
                    help="clean : retire aussi voix et ligne mélodique de 'other' ; simple : tout sauf la piste mélodique")
    ap.add_argument("--player", action="store_true", help="génère le lecteur d'entraînement HTML (3 designs) + zip")
    ap.add_argument("--site", action="store_true", help="ajoute le résultat à la bibliothèque du site (site/) puis la reconstruit")
    ap.add_argument("--site-id", help="identifiant du morceau dans le site (partition.html?id=...)")
    ap.add_argument("--title", help="titre du morceau (par défaut : titre YouTube)")
    ap.add_argument("--solo-label", help="nom de la piste solo dans le lecteur, ex. 'Solo (violon IA)'")
    ap.add_argument("--duo", default="auto", choices=["auto", "off"],
                    help="auto : si la piste solo contient deux violons (deux lignes indépendantes pendant 2 mesures ou plus), "
                         "écrit une partie de Violon II et sépare l'audio en deux pistes ; off : une seule ligne")
    ap.add_argument("--key-plan", default="off", choices=["auto", "off"],
                    help="auto : détecte les modulations (sections d'au moins 8 mesures) et écrit les changements d'armure")
    ap.add_argument("--motifs", default="auto", choices=["auto", "off"],
                    help="auto : un rythme de mesure qui revient (>= 3 fois) sert de modèle ; les mesures proches sont réécrites "
                         "selon ce motif et ses notes brèves finales liées (legato)")
    ap.add_argument("--solo3-label", help="nom de la piste du 3e violon (par défaut « Violon III »)")
    ap.add_argument("--transcripteur", default="basicpitch", choices=["basicpitch", "crepe", "pyin", "mt3"],
                    help="transcription des notes : basicpitch (polyphonique, défaut), crepe / pyin (f0 monophonique + attaques), mt3")
    ap.add_argument("--roles-depuis", type=int, default=0, help="à partir de cette mesure, passages à plusieurs violons : "
                    "Violon II en croches régulières, Violon III en contretemps (0 = non)")
    ap.add_argument("--rythme", default="auto", choices=["auto", "tango", "off"],
                    help="a priori rythmique : tango = figures d'accompagnement (demi-soupir + 3 doubles, 4 doubles en levée, contretemps) "
                         "écrites là où les attaques audio les confirment ; auto = tango si --style tango")
    ap.add_argument("--voix", type=int, default=3, choices=[2, 3], help="nombre maximal de violons dans la piste solo (défaut 3)")
    ap.add_argument("--solo2-label", help="nom de la piste du 2e violon dans le lecteur (par défaut « Violon II »)")
    ap.add_argument("--out", help="dossier de sortie")
    ap.add_argument("--reuse", action="store_true", help="réutilise le téléchargement et la séparation déjà faits dans --out")
    a = ap.parse_args()
    global TRANSCRIBER; TRANSCRIBER = a.transcripteur
    prof_path = Path(a.profil) if a.profil else (ROOT / "profiles" / f"{a.style}.json" if a.style else None)
    prof = None
    if prof_path and prof_path.exists():
        prof = load_profile(prof_path)
        log(f"Profil d'apprentissage : {prof_path} ({len(prof.get('sessions', []))} correction(s) apprise(s))")
    elif a.profil:
        sys.exit(f"Profil introuvable : {prof_path}")
    if a.grid is None:
        a.grid = int((prof or {}).get("suggestions", {}).get("grille", 4))

    src = a.source
    is_url = re.match(r"https?://", src) is not None
    start = parse_time(a.start) if a.start else (start_from_url(src) if is_url else None) or 0.0
    end = parse_time(a.end) if a.end else start + 120
    meta = {}
    out = Path(a.out or f"stem2violin_out_{int(start)}").resolve()
    work = out / "_work"; work.mkdir(parents=True, exist_ok=True)

    pad = 4.0
    raw = work / "source.m4a"
    cached = a.reuse and (work / "meta.json").exists() and (work / "segment.wav").exists()
    offset0 = max(0, start - pad) if is_url else 0.0     # where the downloaded/raw file starts on the source timeline
    if cached:
        meta = json.loads((work / "meta.json").read_text())
        log("Réutilisation du segment déjà téléchargé")
    elif is_url:
        log("Métadonnées YouTube…")
        meta, client = yt_meta(src)
        end = min(end, meta.get("duration") or end)
        log(f"« {meta.get('title')} » ({fmt_time(meta.get('duration', 0))})")
        log(f"Téléchargement de {fmt_time(start)} à {fmt_time(end)}…")
        mark("telechargement")
        yt_download(src, client, max(0, start - pad), end + pad, raw)
        cand = sorted(work.glob("source.*"))
        raw = cand[0]
        offset0 = max(0, start - pad)
    else:
        raw = Path(src); offset0 = 0.0
        meta = {"title": Path(src).stem}
    title_src = a.title or meta.get("title") or "Sans titre"

    seg = work / "segment.wav"
    if not cached:
      (work / "meta.json").write_text(json.dumps({k: meta.get(k) for k in ("title", "webpage_url", "duration", "chapters", "uploader")}))
      run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-ss", f"{start - offset0:.3f}", "-to", f"{end - offset0 + pad:.3f}",
         "-vn", "-ac", "2", "-ar", "44100", str(seg)])
    silent_spans = digital_silences(seg, start)
    for s0, s1 in silent_spans:
        log(f"ATTENTION : silence numérique dans la source de {fmt_time(s0)} à {fmt_time(s1)} "
            f"({s1 - s0:.0f} s) — aucune note ne pourra y être transcrite")

    fullstems = work / "sep" / "htdemucs_6s" / "segment"
    if a.stems_dir:
        log(f"Pistes fournies ({a.stems_dir}) : pas de Demucs")
        mark("import_pistes")
        fullstems = import_stems(Path(a.stems_dir), a.stem_map, start - offset0, end - offset0 + pad, fullstems)
    elif not (cached and (fullstems / "guitar.wav").exists()):
        log("Séparation des pistes (Demucs htdemucs_6s, 1 à 3 min)…")
        mark("demucs")
        fullstems = separate(seg, work / "sep")
    mark("tempo")
    log("Tempo et mesures…")
    bpm, period, beats, downbeats, tracked = beat_grid(fullstems)
    db0 = next((d for d in downbeats if d >= -0.05), downbeats[0])
    if a.downbeat is not None:
        db0 = a.downbeat
    nbars = int(((end - start) - db0) / (4 * period))
    bmap = beat_map(tracked, db0, period, 4 * nbars)
    while nbars > 1 and db0 + bmap[4 * nbars] > (end - start) + 0.02:
        nbars -= 1
    bmap = bmap[:4 * nbars + 1]
    clip_dur = float(bmap[-1])
    drift = float(np.max(np.abs(bmap - np.arange(len(bmap)) * period)))
    log(f"Pulsation suivie temps par temps (écart max à une pulsation fixe : {drift * 1000:.0f} ms)")
    log(f"Tempo ≈ {bpm:.1f} BPM, 1re mesure à {fmt_time(start + db0)} (+{db0:.2f}s), {nbars} mesures")
    stemdir = trim_stems(fullstems, db0, clip_dur, work / "stems_clip")
    mark("basic_pitch_et_transcription")
    log("Analyse des pistes et transcription (Basic Pitch)…")
    sel, info, notes = choose_stems(stemdir, a.stem, clip_dur)
    for s_ in ("other",):
        if s_ not in notes or (not notes[s_] and info[s_]["rms_db"] > -50):
            notes[s_] = transcribe(stemdir / f"{s_}.wav")
    for s_ in STEMS:
        log(f"   {s_:7s} {info[s_]['rms_db']:6.1f} dB  score mélodique={info[s_].get('melodic_score', '-')}")
    cands = [c for c in (AUTO_LEAD_CANDIDATES if a.stem == "auto" else sel) if notes.get(c)]
    barlen = 4 * period
    bar_choice, primary, bar_scores = per_bar_lead(notes, cands, nbars, barlen, clip_dur)
    used = sorted({c for c in bar_choice if c})
    sections = []
    for i, c in enumerate(bar_choice):
        if sections and sections[-1][2] == c:
            sections[-1][1] = i + 1
        else:
            sections.append([i + 1, i + 1, c])
    log("Mélodie par mesures : " + ", ".join(f"m.{x}-{y} {c or 'silence'}" for x, y, c in sections))

    gated = {s_: gate_notes(lead_notes(notes[s_], s_), work / f"gate_{s_}.wav", stemdir, [s_], period) for s_ in used}
    mel_notes = [n for s_, ns in gated.items() for n in ns
                 if 0 <= int(n[0] // barlen) < nbars and bar_choice[int(n[0] // barlen)] == s_]
    drones = frozenset(drone_pitches(notes[primary], clip_dur))
    n_raw = len(mel_notes)
    lead_wav = work / f"gate_{primary}.wav"
    if len(used) > 1:
        lead_wav = work / "lead_mix.wav"; mix_stems(stemdir, used, lead_wav)
    # two violins? (two independent lines in the lead stem for >= 2 bars)
    duo_runs, trio_runs, duo_span, v1_in, v2_notes, v3_notes = [], [], None, [], [], []
    raw_v = {2: [], 3: []}
    if a.duo == "auto":
        duo_runs, _, trio_runs = duo_bars(lead_wav, bmap, nbars)
        if a.voix < 3:
            trio_runs = []
    if duo_runs:
        b0, b1 = duo_runs[0][0], duo_runs[-1][1]
        duo_span = (float(bmap[4 * (b0 - 1)]) - 0.15, float(bmap[4 * b1]))
        inside = [n for n in mel_notes if duo_span[0] <= n[0] < duo_span[1]]
        before = [n for n in mel_notes if n[0] < duo_span[0]]
        prev_pitch = max(before, key=lambda n: n[0])[2] if before else None
        tbar = lambda b_: float(bmap[min(4 * b_, len(bmap) - 1)])
        def nv_at(t, _tr=trio_runs):
            bar = int(np.searchsorted(bmap, t, side="right") - 1) // 4 + 1
            return 3 if any(x <= bar <= y for x, y in _tr) else 2
        runs_t = [(tbar(x - 1) - 0.15, tbar(y)) for x, y in duo_runs]
        v1_in, v2_raw, v3_raw = assign_voices_n(inside, nv_at, runs_t, prev_pitch)
        raw_v = {2: list(v2_raw), 3: list(v3_raw)}
        mel_notes = [n for n in mel_notes if not (duo_span[0] <= n[0] < duo_span[1])] + v1_in
        v2_notes = clean_lead_notes(v2_raw, lead_wav, drones=drones)
        v3_notes = clean_lead_notes(v3_raw, lead_wav, drones=drones) if trio_runs else []
        log("Deux violons détectés : mesures " + ", ".join(f"{x}–{y}" for x, y in duo_runs) +
            f" -> partie de Violon II ({len(v2_notes)} notes) et piste audio séparée")
        if v3_notes:
            log("Trois violons : mesures " + ", ".join(f"{x}–{y}" for x, y in trio_runs) + f" -> Violon III ({len(v3_notes)} notes)")
    raw_v[1] = list(mel_notes)
    mel_notes = clean_lead_notes(mel_notes, lead_wav, drones=drones)
    log(f"Nettoyage des notes : {n_raw} -> {len(mel_notes)} (fragments de notes tenues fusionnés, harmoniques et petites notes parasites retirées)")
    line = extract_melody(mel_notes, clip_dur, period, sub=a.grid, drones=drones, bmap=bmap)
    line = smooth_octaves(line)
    line2 = smooth_octaves(extract_melody(v2_notes, clip_dur, period, sub=a.grid, drones=drones, bmap=bmap)) if v2_notes else []
    line3 = smooth_octaves(extract_melody(v3_notes, clip_dur, period, sub=a.grid, drones=drones, bmap=bmap)) if v3_notes else []
    tango_rep = None
    if a.rythme == "tango" or (a.rythme == "auto" and (a.style or "").lower() == "tango"):
        log("A priori rythmique tango (figures d'accompagnement, contretemps), vérifié sur les attaques audio…")
        ons = audio_onsets(lead_wav); tango_rep = {}
        line, tango_rep["violon_I"] = tango_rhythm(line, a.grid, raw_v[1], ons, bmap, period, log_=log, name="Violon I")
        if line2:
            line2, tango_rep["violon_II"] = tango_rhythm(line2, a.grid, raw_v[2], ons, bmap, period, log_=log, name="Violon II")
        if line3:
            line3, tango_rep["violon_III"] = tango_rhythm(line3, a.grid, raw_v[3], ons, bmap, period, log_=log, name="Violon III")
    slurs, slurs2, slurs3, motifs, motif_bars, motif_bars2 = [], [], [], [], [], []
    if a.motifs == "auto":
        line, slurs, motifs, motif_bars = motif_pass(line, a.grid, nbars)
        if line2:
            line2, slurs2, _, motif_bars2 = motif_pass(line2, a.grid, nbars, motifs=motifs)
        if line3:
            line3, slurs3, _, _ = motif_pass(line3, a.grid, nbars, motifs=motifs)
        pass
    roles_rep = None
    if a.roles_depuis and (line2 or line3) and a.grid == 4:
        log(f"Rôles rythmiques des violons d'accompagnement à partir de la mesure {a.roles_depuis} (II = croches, III = contretemps)…")
        ons_r = audio_onsets(lead_wav); roles_rep = {}
        multi = sorted({b for x, y in duo_runs for b in range(x, y + 1) if b >= a.roles_depuis})
        tri = sorted({b for x, y in trio_runs for b in range(x, y + 1) if b >= a.roles_depuis})
        if line2:
            line2, roles_rep["violon_II"] = voice_role_rhythm(line2, raw_v[2], ons_r, bmap, period, multi, "croches", log_=log, name="Violon II", force=set(tri))
            keys2 = {k for k, _, _ in line2}; slurs2 = [x for x in slurs2 if x[0] in keys2 and x[1] in keys2 and not any(16 * (b - 1) <= x[0] < 16 * b for b in multi)]
        if line3:
            line3, roles_rep["violon_III"] = voice_role_rhythm(line3, raw_v[3], ons_r, bmap, period, tri, "contretemps", log_=log, name="Violon III", force=set(tri))
            keys3 = {k for k, _, _ in line3}; slurs3 = [x for x in slurs3 if x[0] in keys3 and x[1] in keys3 and not any(16 * (b - 1) <= x[0] < 16 * b for b in tri)]
    if a.motifs == "auto" and motifs:
        if motifs:
            q = 1 / a.grid
            log("Motif récurrent : " + " ; ".join("+".join(f"{d * q:g}" for _, d in m) + " temps" for m in motifs) +
                f" -> mesures réécrites selon le motif : {motif_bars}" + (f" (Violon II : {motif_bars2})" if motif_bars2 else ""))
    tonic_pc, mode = detect_key(stemdir, used, line)
    key_shift, new_pc, new_mode = choose_key(tonic_pc, mode, line, a.key)
    line = [(k, d, p + key_shift) for k, d, p in line]
    line, shift = violin_transpose(line, a.transpose)
    if line2:
        line2 = [(k, d, p + key_shift + shift) for k, d, p in line2]
    if line3:
        line3 = [(k, d, p + key_shift + shift) for k, d, p in line3]
    log(f"Tonalité d'origine {PC_NAMES[tonic_pc]} {mode} -> {PC_NAMES[new_pc]} {new_mode} "
        f"(audio et partition transposés de {key_shift:+d} demi-ton(s)) ; octave partition {shift:+d}")
    log(f"{len(line)} notes retenues")
    kplan = None
    if a.key_plan == "auto":
        kp0 = key_plan(stemdir, used, [(k, d, p - key_shift - shift) for k, d, p in line], a.grid, nbars, bmap, tonic_pc, mode)
        kplan = [(b, (pc + key_shift) % 12, md) for b, pc, md in kp0]
        log("Plan tonal : " + " -> ".join(f"m. {b + 1} {PC_NAMES[pc]} {'mineur' if md == 'minor' else 'majeur'}" for b, pc, md in kplan))
        if (kplan[0][1], kplan[0][2]) != (new_pc, new_mode):
            log(f"   tonalité de départ corrigée par le plan tonal : {PC_NAMES[new_pc]} {new_mode} -> "
                f"{PC_NAMES[kplan[0][1]]} {kplan[0][2]} (harmonie de tout l'accompagnement, pas seulement la basse)")
            new_pc, new_mode = kplan[0][1], kplan[0][2]

    piece = re.split(r"\s*[|(]", title_src)[0].strip() or title_src
    title = f"Partie de violon — {piece}"
    KEY_FR = {"C": "do", "C#": "do♯", "D": "ré", "E-": "mi♭", "E": "mi", "F": "fa", "F#": "fa♯", "G": "sol",
              "A-": "la♭", "A": "la", "B-": "si♭", "B": "si"}
    key_fr = f"{KEY_FR[PC_NAMES[new_pc]]} {'mineur' if new_mode == 'minor' else 'majeur'}"
    tr = (f" — transposé {'+' if key_shift > 0 else '−'}{abs(key_shift)} demi-ton{'s' if abs(key_shift) > 1 else ''} ({key_fr})") if key_shift else ""
    wurl = meta.get("webpage_url") or src
    m_id = re.search(r"v=([\w-]{11})", wurl or "")
    short = f"youtu.be/{m_id.group(1)}" if m_id else Path(str(wurl)).name
    src_note = (f"Transcription automatique — {short}, "
                f"{fmt_time(start + db0)}–{fmt_time(start + db0 + clip_dur)}{tr}")
    ks = build_score(line, a.grid, bpm, nbars, title, src_note, out / "violin_part", new_pc, new_mode,
                     response_text=a.response_text, line2=line2, slurs=slurs, slurs2=slurs2, kplan=kplan,
                     line3=line3, slurs3=slurs3)
    mark("musescore")
    log("Mise en page MuseScore…")
    render(out / "violin_part.musicxml", out / "violin_part.pdf", out / "violin_part.png", hide_empty=bool(line2))
    for nm in ("violon_I", "violon_II", "violon_III"):
        if line2 and (nm != "violon_III" or line3):
            render(out / f"{nm}.musicxml", out / f"{nm}.pdf", out / f"{nm}.png", preview=False)
        else:
            for ext in (".musicxml", ".mscz", ".pdf", ".mss"):
                (out / f"{nm}{ext}").unlink(missing_ok=True)

    mark("separation_ligne_et_mixage")
    log("Pistes audio (solo / accompagnement)…")
    if a.backing == "clean":
        split_lead(stemdir / "other.wav", lead_notes(notes["other"], "other"), work / "other_lead.wav", work / "other_rest.wav")
        solo_stems = sorted((set(used) - {"other"}) | {"vocals"})
        solo_paths = [stemdir / f"{x}.wav" for x in solo_stems] + [work / "other_lead.wav"]
        back_stems = [x for x in STEMS if x not in solo_stems and x != "other"]
        back_paths = [stemdir / f"{x}.wav" for x in back_stems] + [work / "other_rest.wav"]
        solo_desc = solo_stems + ["other (ligne mélodique extraite)"]
        back_desc = back_stems + ["other (bourdon/nappes, ligne mélodique retirée)"]
    else:
        solo_paths = [stemdir / f"{x}.wav" for x in used]
        back_stems = [x for x in STEMS if x not in used]
        back_paths = [stemdir / f"{x}.wav" for x in back_stems]
        solo_desc, back_desc = used, back_stems
    mix_wavs(solo_paths, work / "solo_orig.wav"); mix_wavs(back_paths, work / "backing_orig.wav")
    duo_quality = None
    if line2:
        log("Séparation audio Violon I / Violon II (masques harmoniques guidés par la partition)…")
        v1_span = [n for n in mel_notes if duo_span[0] <= n[0] < duo_span[1]]
        vlist = [v1_span, v2_notes] + ([v3_notes] if line3 else [])
        duo_quality = split_voices_audio_n(work / "solo_orig.wav", vlist, duo_span[0], duo_span[1],
                                           [work / "solo_v1_orig.wav", work / "solo_v2_orig.wav", work / "solo_v3_orig.wav"][:len(vlist)])
        shutil.move(work / "solo_v1_orig.wav", work / "solo_orig.wav")
        log(f"   fuite estimée : {duo_quality}")
    origdir = out / "hauteur_originale"
    if key_shift:                                   # untransposed copies only when the audio is transposed
        origdir.mkdir(exist_ok=True)
        to_mp3(work / "solo_orig.wav", origdir / "solo_track_original.mp3")
        if line2:
            to_mp3(work / "solo_v2_orig.wav", origdir / "solo2_track_original.mp3")
        if line3:
            to_mp3(work / "solo_v3_orig.wav", origdir / "solo3_track_original.mp3")
        to_mp3(work / "backing_orig.wav", origdir / "backing_track_original.mp3")
    elif origdir.exists():
        shutil.rmtree(origdir)
    if key_shift:
        log(f"Transposition audio {key_shift:+d} demi-ton(s) (Rubber Band R3, tempo inchangé)…")
    mark("rubberband")
    pitch_shift(work / "solo_orig.wav", work / "solo.wav", key_shift)
    pitch_shift(work / "backing_orig.wav", work / "backing.wav", key_shift)
    mark("mp3")
    to_mp3(work / "backing.wav", out / "backing_track.mp3")
    to_mp3(work / "solo.wav", out / "solo_track.mp3")
    if line2:
        pitch_shift(work / "solo_v2_orig.wav", work / "solo2.wav", key_shift)
        to_mp3(work / "solo2.wav", out / "solo2_track.mp3")
    else:
        (out / "solo2_track.mp3").unlink(missing_ok=True)
    if line3:
        pitch_shift(work / "solo_v3_orig.wav", work / "solo3.wav", key_shift)
        to_mp3(work / "solo3.wav", out / "solo3_track.mp3")
    else:
        (out / "solo3_track.mp3").unlink(missing_ok=True)
    for old in ("zither_isolated.mp3",):
        (out / old).unlink(missing_ok=True)
    if a.count_in:
        count_in_mp3(work / "backing.wav", out / "backing_track_countin.mp3", period)

    mark(None)
    report = {"title": a.title or meta.get("title"), "solo_label": a.solo_label, "url": meta.get("webpage_url") or src, "duration": meta.get("duration"),
              "chapters": meta.get("chapters"), "excerpt_start": start + db0, "excerpt_end": start + db0 + clip_dur,
              "bars": nbars, "bpm": bpm, "beat_period": period,
              "beat_times": [round(float(x), 4) for x in bmap], "key": str(ks),
              "motifs": [[[o / a.grid, d / a.grid] for o, d in m] for m in motifs], "motif_bars": motif_bars,
              "profil": str(prof_path) if prof else None, "reglages": dict(TUNING),
              "key_plan": [{"mesure": b + 1, "tonique": PC_NAMES[pc], "mode": md} for b, pc, md in kplan] if kplan else None,
              "motif_bars_violon2": motif_bars2,
              "duo_bars": duo_runs, "duo_quality_db": duo_quality, "n_notes_violon2": len(line2),
              "solo2_label": (a.solo2_label or "Violon II") if line2 else None,
              "rythme_tango": tango_rep, "roles_violons": roles_rep,
              "trio_bars": trio_runs if line3 else [], "n_notes_violon3": len(line3),
              "solo3_label": (a.solo3_label or "Violon III") if line3 else None,
              "key_original": f"{PC_NAMES[tonic_pc]} {mode}", "key_shift_semitones": key_shift,
              "score_octave_shift": shift, "lead_by_bar": bar_choice, "lead_sections": sections,
              "bar_scores": bar_scores, "solo_track": solo_desc, "backing_track": back_desc,
              "stem_info": info, "n_notes": len(line), "drone_pcs": sorted(drones),
              "timings_s": dict(TIMINGS), "digital_silences": silent_spans, "audio_seconds": round(clip_dur, 1)}
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    shutil.copy(ROOT / "README.md", out / "README.md") if (ROOT / "README.md").exists() else None
    if a.player:
        log("Lecteur HTML…")
        sys.path.insert(0, str(ROOT / "player"))
        import build_player
        import unicodedata
        slug = unicodedata.normalize("NFKD", piece).encode("ascii", "ignore").decode().lower()
        slug = (re.findall(r"[a-z0-9]+", slug) or ["extrait"])[0]
        files, z = build_player.build(out, f"lecteur_{slug}")
        log("Lecteur :", z)
    if a.site:
        import sitegen
        sitegen.add(out, piece_id=a.site_id, log=log)
        log("Site :", sitegen.build(log=log))
    log("Terminé :", out)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

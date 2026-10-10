"""stem2violin apprendre — apprendre des corrections de Pascal.

Compare la transcription automatique (MusicXML d'origine) et la version corrigée (par ex. téléchargée depuis
l'éditeur du lecteur), classe les corrections mesure par mesure, les cumule dans un profil par style
(profils/<style>.json) et ajuste les réglages de transcription que « stem2violin --style <style> » réutilise.

    stem2violin apprendre original.musicxml corrige.musicxml [--style tango] [--profil p.json] [--rapport r.md]
"""
import argparse, hashlib, json, sys, time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
NOM = ["do", "do♯", "ré", "mi♭", "mi", "fa", "fa♯", "sol", "sol♯", "la", "si♭", "si"]
EPS = 1e-6

# réglages par défaut (doivent rester alignés sur stem2violin.TUNING) et bornes de l'apprentissage
DEFAUTS = {"attaque_flux": 3.0, "attaque_niveau": 2.5, "fenetre_double_attaque": 0.2,
           "penalite_simplicite": 1.0, "motif_min": 3, "motif_tolerance": 0.5}
BORNES = {"attaque_flux": (2.0, 6.0), "attaque_niveau": (1.5, 5.0), "fenetre_double_attaque": (0.1, 0.4),
          "penalite_simplicite": (0.5, 3.0), "motif_min": (2, 4), "motif_tolerance": (0.25, 1.0)}
TYPES_FR = {
    "identique": "mesures identiques", "ecriture": "même musique, écriture changée (liaisons de prolongation, silences)",
    "rythme": "rythme corrigé (mêmes notes)", "octave": "notes changées d'octave", "alteration": "altérations corrigées (±1 demi-ton)",
    "hauteur": "autres hauteurs corrigées", "fusion": "notes fusionnées (fragments d'une même note)",
    "fusion_hauteurs": "notes différentes fusionnées en une", "division": "notes divisées", "supprimee": "notes supprimées",
    "ajoutee": "notes ajoutées", "reecriture": "mesures réécrites", "motif": "mesures mises au motif récurrent",
}
SYMB_FR = {"down-bow": "tiré", "up-bow": "poussé", "fingering": "doigté", "string": "corde", "harmonic": "harmonique",
           "snap-pizzicato": "pizz. Bartók", "staccato": "staccato", "staccatissimo": "staccatissimo", "accent": "accent",
           "tenuto": "tenuto", "breath-mark": "respiration", "fermata": "point d'orgue", "trill-mark": "trille",
           "tremolo": "trémolo", "slur": "liaison", "glissando": "glissando", "arpeggiate": "arpège",
           "wedge:crescendo": "soufflet crescendo", "wedge:diminuendo": "soufflet diminuendo", "pedal": "pédale"}


def nom(midi):
    return "silence" if midi is None else f"{NOM[midi % 12]}{midi // 12 - 1}"


# ---------------------------------------------------------------- lecture MusicXML
def lire(path):
    root = ET.parse(path).getroot()
    names = {sp.get("id"): (sp.findtext("part-name") or sp.get("id")) for sp in root.iter("score-part")}
    parts = []
    for part in root.findall("part"):
        div, barlen, pos = 1.0, 4.0, 0.0
        notes, symb, mlen = [], [], []
        for mi, meas in enumerate(part.findall("measure")):
            start = cur = pos; last_on = cur; far = cur
            for e in meas:
                if e.tag == "attributes":
                    if e.findtext("divisions"):
                        div = float(e.findtext("divisions"))
                    t = e.find("time")
                    if t is not None and t.findtext("beats"):
                        barlen = float(t.findtext("beats")) * 4 / float(t.findtext("beat-type"))
                elif e.tag == "backup":
                    cur -= float(e.findtext("duration")) / div
                elif e.tag == "forward":
                    cur += float(e.findtext("duration")) / div
                elif e.tag == "direction":
                    for dt in e.findall("direction-type"):
                        for c in dt:
                            if c.tag == "dynamics":
                                symb += [(mi, cur - start, "dyn:" + d.tag) for d in c]
                            elif c.tag == "wedge" and c.get("type") != "stop":
                                symb.append((mi, cur - start, "wedge:" + c.get("type")))
                            elif c.tag == "words" and (c.text or "").strip():
                                symb.append((mi, cur - start, "texte:" + c.text.strip()))
                            elif c.tag == "pedal" and c.get("type") == "start":
                                symb.append((mi, cur - start, "pedal"))
                elif e.tag == "note":
                    if e.find("grace") is not None:
                        continue
                    chord = e.find("chord") is not None
                    d = float(e.findtext("duration") or 0) / div
                    on = last_on if chord else cur
                    p = e.find("pitch")
                    midi = None if p is None else 12 * (int(p.findtext("octave")) + 1) + STEP[p.findtext("step")] + int(float(p.findtext("alter") or 0))
                    ties = {t.get("type") for t in e.findall("tie")}
                    nt = e.find("notations")
                    if nt is not None and midi is not None:
                        for c in nt:
                            if c.tag in ("articulations", "technical", "ornaments"):
                                for x in c:
                                    lab = x.tag + (f"={x.text.strip()}" if x.tag in ("fingering", "string") and x.text else "")
                                    symb.append((mi, on - start, lab))
                            elif c.tag in ("slur", "glissando") and c.get("type") == "start":
                                symb.append((mi, on - start, c.tag))
                            elif c.tag in ("fermata", "arpeggiate"):
                                symb.append((mi, on - start, c.tag))
                    if not chord:
                        notes.append({"m": mi, "on": on, "dur": d, "midi": midi, "tie_start": "start" in ties, "tie_stop": "stop" in ties})
                        last_on = on; cur += d
                    far = max(far, cur)
            mlen.append(barlen)
            pos = start + max(barlen, far - start)
        parts.append({"id": part.get("id"), "nom": names.get(part.get("id"), part.get("id")), "notes": notes,
                      "symboles": symb, "mesures": len(mlen), "mlen": mlen})
    return parts


def evenements(notes):
    """Notes écrites -> événements sonores : les liaisons de prolongation sont fondues, les silences consécutifs aussi."""
    ev = []
    for n in notes:
        prev = ev[-1] if ev else None
        if prev and ((n["midi"] is not None and n["tie_stop"] and prev["midi"] == n["midi"]) or (n["midi"] is None and prev["midi"] is None)):
            prev["dur"] += n["dur"]; prev["ecrites"] += 1
            continue
        ev.append({"m": n["m"], "on": n["on"], "dur": n["dur"], "midi": n["midi"], "ecrites": 1})
    return ev


def bar_starts(part):
    out, s = [], 0.0
    for L in part["mlen"]:
        out.append(s); s += L
    return out


# ---------------------------------------------------------------- comparaison
def rythme(ev, s0):
    return tuple((round(e["on"] - s0, 4), round(e["dur"], 4)) for e in ev)


def comparer(po, pc):
    """Différences mesure par mesure entre la partie d'origine po et la partie corrigée pc."""
    eo, ec = evenements(po["notes"]), evenements(pc["notes"])
    bs = bar_starts(pc)
    cnt, details = Counter(), []
    nmes = min(po["mesures"], pc["mesures"])
    by = lambda ev, m: [e for e in ev if e["m"] == m and e["midi"] is not None]
    wr = lambda ns, m: [(round(n["on"], 4), round(n["dur"], 4), n["midi"]) for n in ns if n["m"] == m]
    # motif récurrent de la version corrigée
    rc = Counter(rythme(by(ec, m), bs[m]) for m in range(nmes) if len(by(ec, m)) >= 2)
    motif = next((r for r, c in rc.most_common(1) if c >= 3), None)
    for m in range(nmes):
        O, C = by(eo, m), by(ec, m)
        key = lambda E: [(round(e["on"], 4), round(e["dur"], 4), e["midi"]) for e in E]
        d = {"mesure": m + 1, "types": Counter(), "avant": [f"{nom(e['midi'])}:{e['dur']:g}" for e in O],
             "apres": [f"{nom(e['midi'])}:{e['dur']:g}" for e in C]}
        if motif and len(C) >= 2 and rythme(C, bs[m]) == motif and rythme(O, bs[m]) != motif:
            d["types"]["motif"] += 1
        if key(O) == key(C):
            if wr(po["notes"], m) != wr(pc["notes"], m):
                d["types"]["ecriture"] += 1
            else:
                d["types"]["identique"] += 1
        elif [e["midi"] for e in O] == [e["midi"] for e in C]:
            d["types"]["rythme"] += 1
            d["changees"] = sum(1 for a, b in zip(O, C) if abs(a["on"] - b["on"]) > EPS or abs(a["dur"] - b["dur"]) > EPS)
        elif [(round(e["on"], 4), round(e["dur"], 4)) for e in O] == [(round(e["on"], 4), round(e["dur"], 4)) for e in C]:
            for a, b in zip(O, C):
                if a["midi"] != b["midi"]:
                    iv = b["midi"] - a["midi"]
                    t = "octave" if iv % 12 == 0 else ("alteration" if abs(iv) == 1 else "hauteur")
                    d["types"][t] += 1; d.setdefault("intervalles", []).append(iv)
        else:
            used = set()
            for c in C:
                cov = [i for i, o in enumerate(O) if c["on"] - EPS <= o["on"] < c["on"] + c["dur"] - EPS]
                if len(cov) >= 2:
                    same = all(O[i]["midi"] == c["midi"] for i in cov)
                    d["types"]["fusion" if same else "fusion_hauteurs"] += len(cov) - 1
                    used.update(cov)
                elif len(cov) == 1:
                    used.add(cov[0])
                    if O[cov[0]]["midi"] != c["midi"]:
                        iv = c["midi"] - O[cov[0]]["midi"]
                        d["types"]["octave" if iv % 12 == 0 else ("alteration" if abs(iv) == 1 else "hauteur")] += 1
                        d.setdefault("intervalles", []).append(iv)
                    elif abs(O[cov[0]]["dur"] - c["dur"]) > EPS:
                        d["types"]["rythme"] += 1
                else:
                    # note corrigée qui commence pendant une note d'origine : division, ou ajout sur un silence
                    host = [o for o in O if o["on"] < c["on"] < o["on"] + o["dur"] - EPS]
                    d["types"]["division" if host else "ajoutee"] += 1
            d["types"]["supprimee"] += len([i for i in range(len(O)) if i not in used and not any(
                c["on"] < O[i]["on"] < c["on"] + c["dur"] for c in C)])
            if not d["types"]:
                d["types"]["reecriture"] += 1
        cnt.update(d["types"])
        if set(d["types"]) - {"identique"}:
            details.append(d)
    # symboles ajoutés / retirés
    so, sc = Counter(s[2] for s in po["symboles"]), Counter(s[2] for s in pc["symboles"])
    ajouts, retraits = sc - so, so - sc
    pos_aj = [s for s in pc["symboles"] if s[2] in ajouts]
    n_o = sum(1 for e in eo if e["midi"] is not None)
    n16 = lambda ns: sum(1 for n in ns if n["midi"] is not None and (n["dur"] < 0.5 - EPS or abs((n["on"] * 2) % 1) > EPS))
    return {"partie": pc["nom"], "notes_origine": n_o, "notes_corrigees": sum(1 for e in ec if e["midi"] is not None),
            "mesures_avec_notes": sum(1 for m in range(nmes) if by(eo, m) or by(ec, m)),
            "types": dict(cnt), "details": details, "motif": [list(x) for x in motif] if motif else None,
            "symboles_ajoutes": dict(ajouts), "symboles_retires": dict(retraits),
            "symboles_positions": [{"mesure": m + 1, "temps": round(t + 1, 3), "symbole": s} for m, t, s in pos_aj],
            "derniere_mesure": max((e["m"] + 1 for e in ec if e["midi"] is not None), default=0),
            "doubles_origine": n16(po["notes"]), "doubles_corrigees": n16(pc["notes"])}


# ---------------------------------------------------------------- apprentissage
def ajuster(params, res, log):
    """Règles d'ajustement (pas modestes, bornés) à partir des corrections de cette session."""
    P = dict(params); why = []
    N = max(1, sum(r["notes_origine"] for r in res))
    M = max(1, sum(r["mesures_avec_notes"] for r in res))
    T = Counter()
    for r in res:
        T.update(r["types"])
    def borne(k, v):
        lo, hi = BORNES[k]; v = min(hi, max(lo, v))
        return int(round(v)) if isinstance(DEFAUTS[k], int) else round(v, 3)
    def change(k, v, raison):
        v = borne(k, v)
        if v != P[k]:
            why.append({"parametre": k, "avant": P[k], "apres": v, "raison": raison}); P[k] = v
    rf = (T["fusion"] + 0.5 * T["fusion_hauteurs"]) / N      # notes parasites (harmoniques, vibrato) fondues : à moitié
    rd = T["division"] / N
    if rf > 0.02 and rf >= rd:
        f = 1 + min(0.3, 2 * rf)
        change("attaque_flux", P["attaque_flux"] * f, f"{T['fusion'] + T['fusion_hauteurs']} fragment(s) fusionné(s) à la main (taux pondéré {rf:.0%}) : il faut une attaque plus nette pour couper une note tenue")
        change("attaque_niveau", P["attaque_niveau"] * f, "idem (montée de niveau)")
        change("fenetre_double_attaque", P["fenetre_double_attaque"] + min(0.1, rf), "idem (double attaque en début de note)")
    elif rd > 0.02:
        f = 1 - min(0.2, 2 * rd)
        change("attaque_flux", P["attaque_flux"] * f, f"{T['division']} note(s) divisée(s) à la main ({rd:.0%}) : des attaques réelles étaient fondues")
        change("attaque_niveau", P["attaque_niveau"] * f, "idem (montée de niveau)")
    rr = T["rythme"] / M
    if rr > 0.08:
        change("penalite_simplicite", P["penalite_simplicite"] * (1 + min(0.5, rr)),
               f"rythme corrigé dans {T['rythme']} mesure(s) sur {M} : préférer davantage les positions simples (temps, croche)")
    if T["motif"]:
        change("motif_tolerance", P["motif_tolerance"] + 0.125,
               f"{T['motif']} mesure(s) mises au motif à la main : élargir la tolérance du motif")
    sugg = {}
    d0, d1 = sum(r["doubles_origine"] for r in res), sum(r["doubles_corrigees"] for r in res)
    if d0 >= 4 and d1 <= 0.25 * d0:
        sugg["grille"] = 2
        why.append({"parametre": "grille", "avant": 4, "apres": 2, "raison": f"doubles-croches presque toutes retirées ({d0} → {d1}) : quantifier à la croche"})
    ivs = [iv for r in res for d in r["details"] for iv in d.get("intervalles", []) if iv % 12 == 0]
    if len(ivs) >= 3 and abs(sum(1 if i > 0 else -1 for i in ivs)) >= 0.8 * len(ivs):
        sugg["octave_partition"] = 12 if sum(ivs) > 0 else -12
        why.append({"parametre": "octave_partition", "avant": 0, "apres": sugg["octave_partition"], "raison": f"{len(ivs)} notes déplacées d'octave dans le même sens"})
    return P, sugg, why


def apprendre(orig, corr, profil, style, rapport, enregistrer=True, log=print):
    po, pc = lire(orig), lire(corr)
    if len(po) != len(pc):
        log(f"Attention : {len(po)} partie(s) d'origine, {len(pc)} corrigée(s) ; comparaison des {min(len(po), len(pc))} premières.")
    res = [comparer(a, b) for a, b in zip(po, pc)]
    prof = json.loads(profil.read_text()) if profil.exists() else {"style": style, "parametres": dict(DEFAUTS), "sessions": [],
                                                                     "cumul": {}, "symboles": {}, "suggestions": {}}
    params = {**DEFAUTS, **prof.get("parametres", {})}
    empreinte = hashlib.sha1(Path(orig).read_bytes() + b"\0" + Path(corr).read_bytes()).hexdigest()[:16]
    if any(x.get("empreinte") == empreinte for x in prof.get("sessions", [])):
        # même paire déjà apprise : on ne compte pas deux fois les mêmes corrections
        log("[apprendre] cette correction est déjà dans le profil : réglages inchangés.")
        md = rapport_md(orig, corr, profil, style, res, params, params, [], prof, False)
        rapport.write_text(md, encoding="utf-8")
        return prof, res, md
    newp, sugg, why = ajuster(params, res, log)
    T, S = Counter(prof.get("cumul", {})), Counter(prof.get("symboles", {}))
    tot, sym = Counter(), Counter()
    for r in res:
        tot.update(r["types"]); sym.update(r["symboles_ajoutes"])
    T.update(tot); S.update(sym)
    session = {"date": time.strftime("%Y-%m-%d %H:%M"), "empreinte": empreinte, "original": str(orig), "corrige": str(corr), "corrections": dict(tot),
               "symboles_ajoutes": dict(sym), "ajustements": why}
    prof.update({"style": style, "parametres": newp, "cumul": dict(T), "symboles": dict(S),
                 "suggestions": {**prof.get("suggestions", {}), **sugg}})
    prof["sessions"] = prof.get("sessions", []) + [session]
    # habitudes d'écriture : symboles ajoutés à chaque session -> proposés pour les prochains imports
    prof["habitudes"] = habitudes(res, prof)
    if enregistrer:
        profil.parent.mkdir(parents=True, exist_ok=True)
        profil.write_text(json.dumps(prof, ensure_ascii=False, indent=1))
    md = rapport_md(orig, corr, profil, style, res, params, newp, why, prof, enregistrer)
    rapport.write_text(md, encoding="utf-8")
    return prof, res, md


def habitudes(res, prof):
    out = []
    dyns = sorted({p["symbole"][4:] for r in res for p in r["symboles_positions"] if p["symbole"].startswith("dyn:")})
    if dyns:
        out.append("nuances d'entrée ajoutées : " + ", ".join(dyns))
    for r in res:
        pos = r["symboles_positions"]
        last = r["derniere_mesure"]
        for p in pos:
            if p["symbole"] == "fermata":
                out.append(f"point d'orgue ajouté (mesure {p['mesure']}" + (", fin du morceau)" if p["mesure"] >= last - 1 else ")"))
        motif_slur = r["motif"] is not None
        if motif_slur:
            out.append("motif récurrent " + " + ".join(f"{d:g}" for _, d in r["motif"]) + " temps (liaison sur les notes brèves)")
        if any(p["symbole"] in ("down-bow", "up-bow") for p in pos):
            out.append("coups d'archet notés au début des phrases")
    return sorted(set(out))


def rapport_md(orig, corr, profil, style, res, avant, apres, why, prof, enregistrer):
    L = [f"# Apprentissage des corrections — style « {style} »", "",
         f"- Transcription d'origine : `{orig}`", f"- Version corrigée : `{corr}`",
         f"- Profil : `{profil}`" + ("" if enregistrer else " (non enregistré : --essai)") + f" — {len(prof['sessions'])} session(s) cumulée(s)", ""]
    for r in res:
        L += [f"## {r['partie']} — {r['notes_origine']} notes → {r['notes_corrigees']}", ""]
        types = {k: v for k, v in r["types"].items() if k != "identique"}
        L.append(f"Mesures inchangées : {r['types'].get('identique', 0)}." + ("" if types else " Aucune correction de notes."))
        if types:
            L += ["", "| correction | nombre |", "|---|---|"] + [f"| {TYPES_FR.get(k, k)} | {v} |" for k, v in sorted(types.items(), key=lambda x: -x[1])]
        if r["details"]:
            L += ["", "| mesure | corrections | avant | après |", "|---|---|---|---|"]
            for d in r["details"]:
                L.append(f"| {d['mesure']} | {', '.join(TYPES_FR.get(k, k) for k in d['types'] if k != 'identique')} | {' '.join(d['avant']) or '—'} | {' '.join(d['apres']) or '—'} |")
        if r["symboles_ajoutes"]:
            L += ["", "Symboles ajoutés : " + ", ".join(f"{SYMB_FR.get(k.split('=')[0], k.replace('dyn:', 'nuance ').replace('texte:', ''))}{('=' + k.split('=')[1]) if '=' in k else ''} ×{v}" for k, v in r["symboles_ajoutes"].items()) + "."]
            L += ["", "| mesure | temps | symbole |", "|---|---|---|"] + [f"| {p['mesure']} | {p['temps']:g} | {p['symbole']} |" for p in r["symboles_positions"]]
        if r["symboles_retires"]:
            L += ["", "Symboles retirés : " + ", ".join(f"{k} ×{v}" for k, v in r["symboles_retires"].items()) + "."]
        if r["motif"]:
            L += ["", "Motif récurrent de la version corrigée : " + " + ".join(f"{d:g}" for _, d in r["motif"]) + " temps."]
        L.append("")
    L += ["## Réglages de transcription", ""]
    if why:
        L += ["| réglage | avant | après | pourquoi |", "|---|---|---|---|"] + [f"| `{w['parametre']}` | {w['avant']} | {w['apres']} | {w['raison']} |" for w in why]
    else:
        L.append("Aucun réglage modifié : les corrections de cette session sont des ajouts d'interprétation (coups d'archet, nuances, "
                 "doigtés) ou trop peu nombreuses pour justifier un changement (seuils : > 2 % de notes fusionnées/divisées, "
                 "> 8 % de mesures au rythme corrigé, 3 changements d'octave dans le même sens).")
    L += ["", "Réglages du profil : " + ", ".join(f"`{k}` = {v}" for k, v in apres.items()) + ".",
          "", f"Prochain import : `stem2violin … --style {style}` (ou `--profil {profil}`).", ""]
    if prof.get("habitudes"):
        L += ["## Habitudes relevées (propositions pour les prochaines partitions)", ""] + [f"- {h}" for h in prof["habitudes"]] + [""]
    if prof.get("cumul"):
        L += ["## Cumul du profil", "", ", ".join(f"{TYPES_FR.get(k, k)} : {v}" for k, v in prof["cumul"].items()) + "."]
    return "\n".join(L) + "\n"


def cli(argv):
    ap = argparse.ArgumentParser(prog="stem2violin apprendre",
                                 description="Apprendre des corrections : compare la transcription et la version corrigée, met à jour le profil du style.")
    ap.add_argument("original", help="MusicXML produit par stem2violin (violin_part.musicxml)")
    ap.add_argument("corrige", help="la même partition corrigée (éditeur du lecteur, MuseScore…)")
    ap.add_argument("--style", default="general", help="style du morceau (tango, sitar…) -> profils/<style>.json")
    ap.add_argument("--profil", help="fichier profil JSON (par défaut profils/<style>.json)")
    ap.add_argument("--rapport", help="rapport Markdown (par défaut : <corrige>_apprentissage.md)")
    ap.add_argument("--essai", action="store_true", help="analyser sans modifier le profil")
    a = ap.parse_args(argv)
    o, c = Path(a.original), Path(a.corrige)
    for f in (o, c):
        if not f.exists():
            sys.exit(f"Fichier introuvable : {f}")
    profil = Path(a.profil) if a.profil else ROOT / "profiles" / f"{a.style}.json"
    rapport = Path(a.rapport) if a.rapport else c.with_name(c.stem + "_apprentissage.md")
    prof, res, md = apprendre(o, c, profil, a.style, rapport, enregistrer=not a.essai)
    print(md)
    print(f"[apprendre] rapport : {rapport}" + ("" if a.essai else f" · profil : {profil}"))
    return 0


if __name__ == "__main__":
    sys.exit(cli(sys.argv[1:]))

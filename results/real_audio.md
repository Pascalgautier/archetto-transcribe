# Comparaison des transcripteurs : Basic Pitch, CREPE, pYIN, MT3

Pistes : Violines de Sangre (piste violons « lead », puis Violon I / II / III séparés) et premier tango (Violon I).
Même post-traitement pour tous : nettoyage, quantification à la double-croche sur la carte des temps, motifs, a priori rythmique tango.
Les mesures sont calculées par rapport aux attaques détectées dans l'audio (librosa, ±50 ms) ; « fin » = écart à la prochaine attaque ou à la prochaine chute d'énergie.
`tools/compare_transcripteurs.py comparaison_transcription` ; option du pipeline : `--transcripteur basicpitch|crepe|pyin|mt3`.

| piste | moteur | temps (s) | notes brutes | F1 attaques ±50 ms (préc./rappel) | erreur fin méd. (ms) | fragmentation | sauts d'octave | F1 après nettoyage | notes partition |
|---|---|---|---|---|---|---|---|---|---|
| vds_lead | basicpitch | 5.4 | 1415 | 0.664 (0.517/0.925) | 117.6 | 0.063 | 0.448 | 0.548 | 517 |
| vds_lead | crepe | 148.1 | 542 | 0.556 (0.476/0.668) | 74.3 | 0.018 | 0.226 | 0.518 | 353 |
| vds_lead | pyin | 47.3 | 589 | 0.526 (0.435/0.668) | 92.9 | 0.012 | 0.044 | 0.48 | 172 |
| vds_lead | mt3 | 93.4 | 1850 | 0.284 (0.231/0.37) | 65.8 | 0.051 | 0.261 | 0.729 | 371 |
| vds_violon_I | basicpitch | 2.7 | 905 | 0.555 (0.439/0.753) | 166.0 | 0.122 | 0.366 | 0.529 | 431 |
| vds_violon_I | crepe | 149.2 | 519 | 0.523 (0.445/0.634) | 68.5 | 0.023 | 0.232 | 0.516 | 341 |
| vds_violon_I | pyin | 47.5 | 601 | 0.546 (0.438/0.726) | 92.9 | 0.012 | 0.04 | 0.5 | 187 |
| vds_violon_I | mt3 | 153.4 | 1579 | 0.19 (0.147/0.269) | 72.8 | 0.042 | 0.431 | 0.516 | 435 |
| vds_violon_II | basicpitch | 2.2 | 277 | 0.769 (0.686/0.874) | 74.4 | 0.152 | 0.399 | 0.784 | 137 |
| vds_violon_II | crepe | 153.2 | 208 | 0.806 (0.76/0.858) | 12.6 | 0.0 | 0.043 | 0.763 | 134 |
| vds_violon_II | pyin | 46.3 | 237 | 0.795 (0.7/0.918) | 11.6 | 0.008 | 0.021 | 0.611 | 139 |
| vds_violon_II | mt3 | 52.8 | 53 | 0.025 (0.057/0.016) | 235.5 | 0.0 | 0.115 | 0.332 | 46 |
| vds_violon_III | basicpitch | 2.4 | 233 | 0.859 (0.845/0.872) | 52.1 | 0.065 | 0.237 | 0.877 | 170 |
| vds_violon_III | crepe | 483.7 | 251 | 0.785 (0.713/0.872) | 18.8 | 0.004 | 0.02 | 0.89 | 172 |
| vds_violon_III | pyin | 47.3 | 269 | 0.806 (0.71/0.932) | 34.8 | 0.015 | 0.019 | 0.747 | 175 |
| vds_violon_III | mt3 | 43.7 | 107 | 0.043 (0.065/0.032) | 101.9 | 0.009 | 0.085 | 0.479 | 78 |
| tango1_violon_I | basicpitch | 1.8 | 183 | 0.458 (0.404/0.527) | 142.6 | 0.286 | 0.148 | 0.546 | 69 |
| tango1_violon_I | crepe | 46.9 | 144 | 0.835 (0.792/0.884) | 23.2 | 0.0 | 0.021 | 0.553 | 72 |
| tango1_violon_I | pyin | 15.4 | 153 | 0.726 (0.66/0.806) | 34.8 | 0.007 | 0.02 | 0.501 | 71 |
| tango1_violon_I | mt3 | 31.3 | 124 | 0.243 (0.29/0.209) | 160.8 | 0.163 | 0.065 | 0.375 | 76 |
Images côte à côte (mesures 8–24) : `cote_a_cote_<piste>_m8-24.png`.

## À lire honnêtement
- **Basic Pitch** : très bon rappel des attaques (repère presque toutes les attaques), mais beaucoup de notes en trop (polyphonique : harmoniques,
  doubles), fins de notes imprécises (70–170 ms), notes tenues coupées en morceaux et nombreux sauts d'octave. Rapide (2–5 s).
- **CREPE** (torchcrepe, modèle full) : f0 monophonique + segmentation aux attaques. Meilleures fins de notes (12–75 ms), quasi pas de fragmentation,
  peu d'erreurs d'octave sur les pistes d'un seul violon ; meilleur F1 d'attaques sur le premier tango (0,84 contre 0,46) et sur Violon II.
  Moins bon sur la piste « lead » où plusieurs violons jouent ensemble (il ne suit qu'une voix). Lent sur CPU (≈ 2,5 min pour 3 min d'audio).
  Note : le décodeur viterbi de torchcrepe donnait une périodicité nulle sur ces pistes ; décodeur weighted_argmax utilisé.
- **pYIN** (librosa) : proche de CREPE, un peu moins précis sur les fins, plus rapide (≈ 45 s), un peu plus de fausses notes courtes.
- **MT3** (Magenta, point de contrôle « mt3 », JAX CPU, environnement séparé) : installable et utilisable (30–150 s par piste),
  mais décevant ici : il répartit le violon sur plusieurs « instruments », invente beaucoup de notes sur les pistes à plusieurs violons et en rate
  beaucoup sur Violon II/III ; ses attaques semblent décalées d'environ 60 ms (avec un décalage constant compensé, la précision du premier tango
  passerait de 0,29 à ≈ 0,55, toujours sous CREPE). Pas retenu.

## Recommandation
**CREPE pour les hauteurs et les durées, sur les pistes d'un seul violon** (après séparation en Violon I/II/III), en gardant Basic Pitch pour
la détection des passages à plusieurs violons (où il faut du polyphonique). pYIN est une bonne solution de repli rapide.
La version publiée sur le site reste Basic Pitch pour l'instant ; les autres versions sont proposées dans le Labo du lecteur.

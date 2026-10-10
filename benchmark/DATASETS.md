# Datasets

| dataset | content | ground truth | license | status |
|---|---|---|---|---|
| `synthetic-archetto` (this repo, `benchmark/data/`) | 4 violin lines (Violines de Sangre Vl. I/II/III, first tango Vl. I) re-synthesised with `benchmark/synth.py` (seeded, deterministic) | exact (rendered from the score) | Apache-2.0 (our scores) | **included** |
| Archetto pieces, real audio + Pascal-corrected scores | the stems of our pieces + scores corrected in the player editor | human-corrected | to be decided by Pascal (ElevenLabs-generated audio: check ElevenLabs terms before publishing) | **not yet available** — no corrected score exists yet; `archetto_transcribe.learn` will produce them |
| URMP (Univ. Rochester) | 44 ensemble pieces, separate violin tracks, note annotations | human (note level) | no license stated in the official docs (registration asked); a Kaggle mirror says CC0, which does not establish the original license | **not downloaded** — ask the authors |
| Bach10 | 10 Bach chorales, violin part | aligned notes | no license stated, download form | **not downloaded** |
| Bach Violin Dataset (Dong et al.) | 6.5 h solo violin, score + estimated alignment | estimated alignment (not exact notes) | code MIT, scores public domain, audio licenses vary per recording | possible later for recordings with permissive licenses; alignment is not note-exact |

Caveat: the synthetic ground truth comes from scores produced by our own pipeline (Basic Pitch based). The audio is
re-synthesised from those notes, so no engine sees its own output, but the melodies/rhythms are "transcribable" by
construction. Real-audio results (no exact ground truth, onsets vs detected audio attacks) are in `results/real_audio.md`.

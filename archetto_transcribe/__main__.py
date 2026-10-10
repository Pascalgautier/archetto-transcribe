"""python -m archetto_transcribe AUDIO [options]  (full pipeline: separation, engine, priors, MusicXML/MIDI/PDF)
python -m archetto_transcribe notes AUDIO --engine ours   (notes only, JSON on stdout)
python -m archetto_transcribe learn original.musicxml corrected.musicxml --style tango"""
import sys, json
if len(sys.argv) > 1 and sys.argv[1] == "notes":
    import argparse
    from .engines import transcribe, ENGINES
    ap = argparse.ArgumentParser(prog="archetto_transcribe notes"); ap.add_argument("audio"); ap.add_argument("--engine", default="ours", choices=ENGINES)
    a = ap.parse_args(sys.argv[2:]); json.dump(transcribe(a.audio, a.engine), sys.stdout)
elif len(sys.argv) > 1 and sys.argv[1] == "learn":
    from . import learn; sys.argv = [sys.argv[0]] + sys.argv[2:]; learn.main() if hasattr(learn, "main") else learn.cli(sys.argv[1:])
else:
    from . import pipeline; pipeline.main()

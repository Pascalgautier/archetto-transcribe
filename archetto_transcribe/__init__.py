"""archetto-transcribe — violin-oriented transcription engine (separation, f0/onset engines, held-note merging,
motifs, style priors, voice roles, learning from corrections). Extracted from the Archetto project."""
__version__ = "0.1.0.dev0"
from .engines import transcribe, ENGINES, ours  # noqa: F401

"""COMET neural MT evaluation metric wrapper.

Uses Unbabel's COMET (wmt22-comet-da) to score translations. COMET is a
learned metric that correlates much better with human judgements than BLEU.

Usage:
    from src.comet_eval import score_comet
    result = score_comet(sources, hypotheses, references)
    # result.system_score  -> float (corpus-level)
    # result.scores        -> list[float] (per-sentence)
"""
from __future__ import annotations

from typing import List, Optional


def score_comet(
    sources: List[str],
    hypotheses: List[str],
    references: List[str],
    model_name: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 16,
    gpus: int = 1,
) -> Optional[dict]:
    """Score translations with COMET.

    Returns dict with 'system_score' (float) and 'scores' (list of floats),
    or None if COMET is not installed.
    """
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        print("[comet] unbabel-comet not installed, skipping COMET scoring")
        return None

    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)

    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]

    output = model.predict(data, batch_size=batch_size, gpus=gpus)
    return {
        "system_score": output.system_score,
        "scores": list(output.scores),
    }

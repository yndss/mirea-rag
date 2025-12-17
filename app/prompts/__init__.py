from pathlib import Path
from functools import lru_cache

_PROMPTS_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=64)
def load_prompt(name: str) -> str:
    """
    Read a prompt template from the prompts directory.
    """
    prompt_path = _PROMPTS_DIR / name
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    return prompt_path.read_text(encoding="utf-8")


__all__ = ["load_prompt"]

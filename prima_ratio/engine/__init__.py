"""prima-ratio decision engine — vendored from SemIf (MIT), Torch paths removed.

Modules: prompts (prompt contract), encoding (slot-verified prompt encoding),
prefix (shared-state prefix), llamacpp (the GGUF engine).
"""

from . import llamacpp
from .encoding import PROMPT_VERSION, encode_prompt
from .prompts import LETTERS, softmax, validate_row

__all__ = ["llamacpp", "PROMPT_VERSION", "encode_prompt", "LETTERS", "softmax", "validate_row"]

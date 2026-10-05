"""Backend-bound, full-prefix-chain identities (not target-only Source keys)."""
from dataclasses import asdict, dataclass
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ExactContext:
    model_revision: str
    tokenizer_template_signature: str
    attention_adapter_signature: str
    position_rope_signature: str
    dtype_layout_backend_signature: str
    namespace: str

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in asdict(self).values()):
            raise ValueError("every exact context field must be explicitly bound")


@dataclass(frozen=True)
class ExactBlockKey:
    context_digest: str
    parent_prefix_digest: str
    token_ids: tuple[int, ...]
    start_position: int
    block_size: int

    def __post_init__(self):
        for value in (self.context_digest, self.parent_prefix_digest):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("unbound context or parent digest")
        if type(self.block_size) is not int or self.block_size <= 0:
            raise ValueError("positive block size required")
        if type(self.start_position) is not int or self.start_position < 0:
            raise ValueError("invalid absolute position")
        if self.start_position % self.block_size:
            raise ValueError("block position is not aligned")
        if not isinstance(self.token_ids, tuple) or len(self.token_ids) != self.block_size:
            raise ValueError("pilot publishes only complete immutable blocks")
        if any(type(t) is not int or t < 0 for t in self.token_ids):
            raise ValueError("invalid token identity")

    @property
    def sha256(self):
        return digest(asdict(self))


def prefix_keys(context: ExactContext, tokens, block_size: int):
    """Exclude the incomplete final block; do not pad or change token IDs."""
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("invalid block_size")
    tokens = tuple(tokens)
    if any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError("invalid tokens")
    context_digest = digest(asdict(context))
    parent = digest({"root": context_digest, "position": 0})
    keys = []
    for start in range(0, len(tokens) - block_size + 1, block_size):
        key = ExactBlockKey(context_digest, parent, tokens[start:start + block_size],
                            start, block_size)
        keys.append(key)
        parent = key.sha256
    return tuple(keys)

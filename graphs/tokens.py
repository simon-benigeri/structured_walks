"""Token vocabulary for graph node labeling.

Words chosen to:
1. Tokenize to a single token in Llama-3.1's tokenizer.
2. Have no obvious semantic correlations with each other.
3. Cover enough words for up to 6x6 = 36 node graphs.
"""

CONCEPT_TOKENS = [
    "apple", "sand", "math", "river", "chair", "salt",
    "king", "dust", "rain", "bold", "fish", "lamp",
    "gold", "wine", "tree", "moon", "rock", "bird",
    "steel", "warm", "dark", "leaf", "drum", "wolf",
    "cake", "silk", "farm", "bell", "star", "pond",
    "gate", "rope", "coal", "mint", "foam", "clay",
]

WEEKDAY_TOKENS = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]


def assign_tokens(num_nodes: int, seed: int | None = None,
                  pool: list[str] | None = None) -> list[str]:
    """Randomly assign concept words to graph nodes.

    The paper "randomly arrange[s] the set of tokens in a grid", so a fixed
    alphabetical or list order would confound any residual semantic
    correlation with graph position.

    Args:
        num_nodes: Number of nodes needing labels.
        seed: Seed for the arrangement. None uses global RNG state.
        pool: Word pool to draw from. Defaults to CONCEPT_TOKENS.

    Returns:
        List of num_nodes words, where index i labels node i.
    """
    import random as _random

    pool = list(CONCEPT_TOKENS if pool is None else pool)
    if num_nodes > len(pool):
        raise ValueError(
            f"Need {num_nodes} labels but the pool has only {len(pool)}."
        )

    rng = _random.Random(seed)
    return rng.sample(pool, num_nodes)


def single_token_pool(tokenizer, pool: list[str] | None = None,
                      prefix: str = " ") -> list[str]:
    """Filter a word pool down to those that are one token for this tokenizer.

    Which words qualify is model-specific, so the pool has to be filtered at
    runtime rather than trusted from the list. Words are checked in the
    space-prefixed form, because that is how they appear mid-sequence.

    Args:
        tokenizer: A HuggingFace tokenizer.
        pool: Candidate words. Defaults to CONCEPT_TOKENS.
        prefix: String prepended before encoding.

    Returns:
        The eligible words, in their original order.
    """
    pool = list(CONCEPT_TOKENS if pool is None else pool)
    ok = verify_single_token(tokenizer, pool, prefix=prefix)
    return [w for w in pool if ok[w]]


def verify_single_token(tokenizer, words: list[str],
                        prefix: str = " ") -> dict[str, bool]:
    """Check which words tokenize to a single token.

    Args:
        tokenizer: A HuggingFace tokenizer.
        words: List of words to check.
        prefix: String prepended before encoding. Defaults to a space, the form
            in which words actually appear mid-sequence; checking the bare word
            can say "single token" for a word that is two tokens in context.

    Returns:
        Dict mapping word -> True if it's a single token, False otherwise.
    """
    results = {}
    for word in words:
        token_ids = tokenizer.encode(prefix + word, add_special_tokens=False)
        results[word] = len(token_ids) == 1
    return results

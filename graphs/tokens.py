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


def verify_single_token(tokenizer, words: list[str]) -> dict[str, bool]:
    """Check which words tokenize to a single token.

    Args:
        tokenizer: A HuggingFace tokenizer.
        words: List of words to check.

    Returns:
        Dict mapping word -> True if it's a single token, False otherwise.
    """
    results = {}
    for word in words:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        results[word] = len(token_ids) == 1
    return results

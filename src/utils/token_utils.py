try:
    import tiktoken  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    tiktoken = None


def get_token_count(prompt: str, model: str = "gpt-4o") -> int:
    """
    Get the number of tokens in a prompt.
    :param prompt: The prompt to count tokens for.
    :param model: The model to use for tokenization. Default is "gpt-4o".
    :return: The number of tokens in the prompt.
    """
    # Fallback if tiktoken is not installed (approx 4 chars per token for English).
    if tiktoken is None:
        return max(1, len(prompt) // 4)

    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(prompt))
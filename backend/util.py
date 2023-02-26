import enum


class OpenAIModelType(enum.Enum):
    """Enum for model types."""
    TEXT_DAVINCI_3_MODEL = "text-davinci-003"


_MAX_TOKEN_COUNT_BY_MODEL = {
    OpenAIModelType.TEXT_DAVINCI_3_MODEL: 4000,
}

TOKENS_PER_WORD_RATE_GPT3 = 4 / 3


def get_max_token_count(model: OpenAIModelType):
    """Get the max token count for a model."""
    return _MAX_TOKEN_COUNT_BY_MODEL.get(model, 2048)

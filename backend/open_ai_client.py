import requests
import openai
import util


class OpenAIClient:
    """A client for the OpenAI GPT-3 Completion API."""

    def complete(self,
                 prompt: str,
                 model=util.OpenAIModelType.TEXT_DAVINCI_3_MODEL,
                 temperature=0.6):
        """Get a response from the GPT-3 Completion API."""
        response_token_count = int(
            util.get_max_token_count(model) -
            len(prompt.split()) * util.TOKENS_PER_WORD_RATE_GPT3)
        if (response_token_count <= 0):
            raise ValueError("Prompt is too long for the model.")
        response = openai.Completion.create(model=model.value,
                                            prompt=prompt,
                                            max_tokens=response_token_count,
                                            temperature=temperature)
        return response.choices[0].text
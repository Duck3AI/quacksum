from typing import Optional

import open_ai_client


class ArticleSummarizer():
    """Summarizes an article using GPT3 completion API."""

    def __init__(self) -> None:
        self._open_ai_client = open_ai_client.OpenAIClient()

    def summarize(self, article: str) -> str:
        """Summarizes an article using GPT3 completion API."""

    def summarize_article_text(self,
                               article_text: str,
                               article_title: Optional[str] = None) -> str:
        """Summarizes an article using GPT3 completion API."""
        chunked_text = article_text
        prmopt = self._generate_prompt(chunked_text, article_title)
        return self._open_ai_client.complete(prompt=prmopt)

    def _generate_prompt(self,
                         chunk_text: str,
                         article_title: Optional[str],
                         summary_length_limit_words=200):
        """Generates a prompt for the OpenAI completion API."""
        return f"""\
Help me summarize the following passages from an article{f" titled {article_title}" if article_title else ""}. \
Provide a brief summary of the main points, ideas and tones presented in these paragraphs as bullet points. \
Keep it under {summary_length_limit_words} words.

Here is an eample for reference:

Passage: 

AFTER DAYBREAK, THE village of Labota begins to shudder with the roar of motorbikes. Thousands of riders in \
canary yellow helmets and dust-stained workwear pack its ramshackle, pothole-ridden main road, in places six \
or seven lanes wide, as it runs along the coast of Indonesia’s Banda Sea. The mass of traffic crawls toward \
the Indonesia Morowali Industrial Park, better known as IMIP, the world’s epicenter for nickel production.

Summary:

* Labota is an busy industrial city near the coast of Indonesia’s Banda Sea which, and it is home to the Indonesia \
Morowali Industrial Park (IMIP), the world's epicenter for nickel production.

Passage: 
{chunk_text}

Summary:
"""

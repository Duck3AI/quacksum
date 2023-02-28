from typing import Generator, Optional

import open_ai_client
import util

_INITIAL_PROMPT_TEMPLATE = """\
Help me summarize the following passages from an article{article_title}. \
Provide a brief summary of the main points, ideas and tones as bullet points. \
Make it between 200 and {summary_length_words} words.

Here is an eample for reference:

Passage: 

AFTER DAYBREAK, THE village of Labota begins to shudder with the roar of motorbikes. Thousands of riders in \
canary yellow helmets and dust-stained workwear pack its ramshackle, pothole-ridden main road, in places six \
or seven lanes wide, as it runs along the coast of Indonesia’s Banda Sea. The mass of traffic crawls toward \
the Indonesia Morowali Industrial Park, better known as IMIP, the world’s epicenter for nickel production.

Summary:

* Labota is a busy industrial city near the coast of Indonesia’s Banda Sea which, and it is home to the Indonesia \
Morowali Industrial Park (IMIP), the world's epicenter for nickel production.

Passage: 
{chunk_text}

Summary:
"""

_INITIAL_PROMPT_LENGTH = len(_INITIAL_PROMPT_TEMPLATE.split())

_CONTINUATION_PROMPT_TEMPLATE = """\
I have a summary of the previous passages of an article{article_title}. \
Help me compose a new summary that includes both the summary of the previous passage, and the next passages. \
Provide a brief summary of the main points, ideas and tones as bullet points. \
Make it between 200 and {summary_length_words} words. Use roughly {previous_chunk_percentage} of the words to \
summarize the previous passage summary, and the rest for the next passage.

Here is an example for reference:

Previous passage summary:
* Labota is a busy industrial city near the coast of Indonesia’s Banda Sea which, and it is home to the Indonesia \
Morowali Industrial Park (IMIP), the world's epicenter for nickel production.

Next passage: 

A decade ago, Labota was a fishing village; today it’s been subsumed into a sprawling city centered around IMIP, a \
$15 billion, 3,000-hectare industrial complex containing steelworks, coal power plants, and manganese processors, with \
its own airport and seaport. Built as a joint venture between Chinese and Indonesian industrial companies, it is at the \
heart of Indonesia’s push to supply the electric vehicle market with nickel, a core component of batteries.

Summary:

* Labota was a fishing village near the coast of Indonesia’s Banda Sea that has turned into a busy industrial city over \
the last decade. 
* It is now home to the Indonesia Morowali Industrial Park (IMIP), a $15 billion industrial complex and \
the world's epicenter for nickel production supplying the electric vehicle market.
* IMIP is a joint venture between Chinese and Indonesian industrial companies

Previous passage summary:
{previous_chunk_text}

Next passage:
{chunk_text}

Summary:
"""

_CONTINUATION_PROMPT_TEMPLATE_LENGTH = len(
    _CONTINUATION_PROMPT_TEMPLATE.split())

_FINAL_PROMPT_TEMPLATE = """\
Write me a short article{article_title} incorporating the following points: \

{talking_points}



Preserve the main points, ideas and tones. Make it between 100 and {summary_length_words} words. \
Do not include a conclusion paragraph.
"""


class ArticleSummarizer():
    """Summarizes an article using GPT3 completion API."""

    def __init__(self) -> None:
        self._open_ai_client = open_ai_client.OpenAIClient()

    def summarize(self, article: str) -> str:
        """Summarizes an article using GPT3 completion API."""

    def summarize_article_text(self,
                               article_text: str,
                               article_title: Optional[str] = None,
                               summary_length_words=400) -> str:
        """Summarizes an article using GPT3 completion API."""
        previous_summary = ""
        total_words_so_far = 0
        for index, chunk in enumerate(
                self._chunk_text(article_text, summary_length_words,
                                 util.OpenAIModelType.TEXT_DAVINCI_3_MODEL)):
            current_chunk_length_words = len(chunk.split())
            if index == 0:
                prompt = self._generate_initial_prompt(chunk, article_title,
                                                       summary_length_words)
            else:
                previous_chunk_percentage = max(
                    int(total_words_so_far /
                        (total_words_so_far + current_chunk_length_words) *
                        100), 2)
                prompt = self._generate_continuation_prompt(
                    chunk, previous_summary, previous_chunk_percentage,
                    article_title, summary_length_words)
            previous_summary = self._open_ai_client.complete(prompt=prompt)
            total_words_so_far += current_chunk_length_words
            print(f"Index: {index}\nSummary: {previous_summary}")

        return self._open_ai_client.complete(
            prompt=self._generate_final_prompt(previous_summary, article_title,
                                               summary_length_words))
        # return previous_summary

    def _chunk_text(self, text: str, summary_length_words,
                    model: util.OpenAIModelType) -> Generator[str, None, None]:
        """Chunks text into smaller chunks."""
        split_by_paragraphs = text.split('\n')
        prompt_and_response_length_words = _CONTINUATION_PROMPT_TEMPLATE_LENGTH + summary_length_words * 2
        max_chunk_length_words = ((util.get_max_token_count(model) -
                                   (prompt_and_response_length_words *
                                    util.TOKENS_PER_WORD_RATE_GPT3)) /
                                  util.TOKENS_PER_WORD_RATE_GPT3)

        current_chunk = []
        current_chunk_length_words = 0
        for paragraph in split_by_paragraphs:
            if paragraph == "\n":
                continue
            stripped_paragraph = paragraph.strip()
            paragraph_length_words = len(stripped_paragraph.split())
            if (current_chunk_length_words + paragraph_length_words >
                    max_chunk_length_words):
                yield "\n".join(current_chunk)
                current_chunk.clear()
                current_chunk_length_words = 0

            current_chunk.append(stripped_paragraph)
            current_chunk_length_words += paragraph_length_words

        if (len(current_chunk)) > 0:
            yield "\n".join(current_chunk)

    def _generate_initial_prompt(self,
                                 chunk_text: str,
                                 article_title: Optional[str],
                                 summary_length_words=200):
        """Generates a prompt for the OpenAI completion API."""
        return _INITIAL_PROMPT_TEMPLATE.format(
            article_title=f" titled {article_title}" if article_title else "",
            summary_length_words=summary_length_words,
            chunk_text=chunk_text)

    def _generate_continuation_prompt(self,
                                      chunk_text: str,
                                      previous_chunk_text: str,
                                      previous_chunk_percentage: int,
                                      article_title: Optional[str],
                                      summary_length_words=200):
        return _CONTINUATION_PROMPT_TEMPLATE.format(
            article_title=f" titled {article_title}" if article_title else "",
            summary_length_words=summary_length_words,
            previous_chunk_percentage=f"{previous_chunk_percentage}%",
            chunk_text=chunk_text,
            previous_chunk_text=previous_chunk_text)

    def _generate_final_prompt(self, summarized_bullets: str,
                               article_title: Optional[str],
                               summary_length_words):
        return _FINAL_PROMPT_TEMPLATE.format(
            summary_length_words=summary_length_words,
            article_title=f" titled {article_title}" if article_title else "",
            talking_points=summarized_bullets)

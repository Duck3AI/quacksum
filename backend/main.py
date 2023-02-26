import openai
import argparse
import pathlib
import summarizers


def _get_article(article_file_path: str):
    article_text = ""
    with open(article_file_path, 'r') as f:
        article_text = f.read()
    return article_text, pathlib.Path(article_file_path).stem


def main():
    parser = argparse.ArgumentParser(
        prog='Summarizer',
        description='Summarizes an article using GPT3 completion API.')
    parser.add_argument('--open_ai_key_file_path',
                        type=str,
                        help='OpenAI API key file')
    parser.add_argument('--article_file_path',
                        type=str,
                        help='Article file to summarize')
    args = parser.parse_args()

    openai.api_key_path = args.open_ai_key_file_path
    article_text, article_title = _get_article(args.article_file_path)
    summarizer = summarizers.ArticleSummarizer()
    summary = summarizer.summarize_article_text(article_text, article_title)
    print(summary)


if __name__ == '__main__':
    main()
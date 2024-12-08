import os

import PyPDF2
import wikipediaapi


def fetch_wikipedia_articles(category: str, lang: str = 'en'):
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent='YourAppName/1.0 (your.email@example.com)',
        language=lang,
    )
    cat = wiki_wiki.page(f'Category:{category}')
    articles = {}

    if cat.exists():
        for page in cat.categorymembers.values():
            if page.ns == wikipediaapi.Namespace.MAIN:
                articles[page.title] = page.text

    return articles


def save_articles(articles, directory: str):
    os.makedirs(directory, exist_ok=True)
    for title, text in articles.items():
        filename = os.path.join(directory, f'{title}.txt')
        with open(filename, 'w', encoding='utf8') as file:
            file.write(text)


def read_pdf_files(directory: str):
    texts = {}
    for file in os.listdir(directory):
        if file.endswith('.pdf'):
            with open(os.path.join(directory, file), 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()
                texts[file] = text
    return texts


if __name__ == '__main__':
    articles = fetch_wikipedia_articles('Formula 1 Drivers')
    save_articles(articles, './data/')
    fia_texts = read_pdf_files('./data/')

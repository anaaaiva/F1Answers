import os

from langchain_community.document_loaders import PyPDFLoader, WikipediaLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_core.documents.base import Document
from settings import PDF_DIR_PATH, WIKI_SEARCH


def load_data(
    wiki_search: list[str] = WIKI_SEARCH, pdf_dir_path: str = PDF_DIR_PATH
) -> list[Document]:
    """
    Loads and returns a list of Documents from specified data sources.

    This function retrieves data from two main sources:
    1. Wikipedia: Searches for the specified queries and loads up to 100 documents per query.
    2. PDFs: Reads all PDF files in the specified directory.

    The data from both sources are combined using a MergedDataLoader to provide a unified list of documents.

    Args:
        wiki_search (list[str]): A list of search queries to fetch data from Wikipedia.
        pdf_dir_path (str): The path to the directory containing PDF files to load.

    Returns:
        list[Document]: A list of Documents containing data loaded from Wikipedia and PDF files.
    """
    wiki_loaders = [
        WikipediaLoader(query=search, load_max_docs=100) for search in wiki_search
    ]

    pdf_files = []
    for file in os.scandir(pdf_dir_path):
        if file.name.endswith('.pdf'):
            pdf_files.append(file.name)

    pdf_loaders = [PyPDFLoader(file) for file in pdf_files]
    all_loader = MergedDataLoader(loaders=wiki_loaders + pdf_loaders)

    return all_loader.load()

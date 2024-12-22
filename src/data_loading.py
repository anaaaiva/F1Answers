import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WikipediaLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
from settings import PDF_DIR_PATH, WIKI_SEARCHS
from utils import CustomEmbeddings


def load_data(
    wiki_searchs: list[str] = WIKI_SEARCHS, pdf_dir_path: str = PDF_DIR_PATH
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
        WikipediaLoader(query=search, load_max_docs=25) for search in wiki_searchs
    ]

    pdf_loaders = [
        PyPDFLoader(file.path)
        for file in os.scandir(pdf_dir_path)
        if file.name.endswith('.pdf')
    ]

    all_loaders = MergedDataLoader(loaders=wiki_loaders + pdf_loaders)

    return all_loaders.load()


def prepare_data(documents: list[Document] = None) -> FAISS:
    embeddings = CustomEmbeddings()

    if os.path.exists('faiss_index'):
        vectorstore = FAISS.load_local(
            'faiss_index', embeddings, allow_dangerous_deserialization=True
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, chunk_overlap=2000
        )
        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local('faiss_index')

    return vectorstore


if __name__ == '__main__':
    documents = load_data()
    prepare_data(documents)

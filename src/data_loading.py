import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WikipediaLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
from settings import FAISS_DIR_PATH, PDF_DIR_PATH, WIKI_SEARCHS
from utils import CustomEmbeddings


def load_data(
    wiki_searchs: list[str] = WIKI_SEARCHS, pdf_dir_path: str = PDF_DIR_PATH
) -> list[Document]:
    """Load data from Wikipedia and PDF files and return as a list of Documents."""
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
    """Prepare data for retrieval by embedding documents with FAISS vector store."""
    embeddings = CustomEmbeddings()

    if os.path.exists(FAISS_DIR_PATH):
        vectorstore = FAISS.load_local(
            FAISS_DIR_PATH, embeddings, allow_dangerous_deserialization=True
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, chunk_overlap=1000
        )
        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(FAISS_DIR_PATH)

    return vectorstore


def format_source(source: Document) -> str:
    """Format the source information for better readability."""
    metadata = source.metadata
    title = metadata.get('title', 'Unknown Title')
    url = metadata.get('source', 'Unknown Source')
    return f'**Title**: {title}\n**Source**: {url}\n\n'


if __name__ == '__main__':
    documents = load_data()
    prepare_data(documents)

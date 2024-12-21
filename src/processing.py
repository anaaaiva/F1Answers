import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
from utils import CustomEmbeddings


def process_data(docs_all: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs_all)
    embeddings = CustomEmbeddings()

    if os.path.exists("faiss_index"):
        vectorstore = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True
        )
    else:
        vectorstore = FAISS.from_documents(splits, embeddings)

        vectorstore.save_local("faiss_index")

    return vectorstore

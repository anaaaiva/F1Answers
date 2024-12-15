import os

import numpy as np
import requests
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from settings import (
    CHATGPT_SYSTEM_PROMPT,
    EMBEDDINGS_BASE_URL,
    EMBEDDINGS_HEADERS,
    EMBEDDINGS_MODEL,
    OPENAI_API_KEY,
)


def generate_embedding(input_text: str) -> np.ndarray:
    data = {'model': EMBEDDINGS_MODEL, 'input': input_text, 'dimensions': 1024}

    try:
        response = requests.post(
            EMBEDDINGS_BASE_URL, headers=EMBEDDINGS_HEADERS, json=data
        )
        response.raise_for_status()
        embedding = response.json()['data'][0]['embedding']
        return np.array(embedding, dtype=np.float32)

    except requests.exceptions.RequestException as e:
        print(f'Error while generating embeddings: {e}')
        return np.array([], dtype=np.float32)


def load_data(file_path: str = './data/regulations.pdf') -> list[dict]:
    if os.path.exists(file_path):
        pdf_loader = PyPDFLoader(file_path=file_path)
        pdf_documents = pdf_loader.load()
        pdf_documents = [pdf_documents[0]]
    else:
        pdf_documents = []

    documents = []
    for doc in pdf_documents:
        embedding = generate_embedding(doc.page_content)
        if embedding.size > 0:
            documents.append({'text': doc.page_content, 'embedding': embedding})

    return documents


def build_index() -> FAISS:
    folder_path = './data'

    if os.path.exists(f'{folder_path}/index.faiss'):
        return FAISS.load_local(
            folder_path=folder_path,
            embeddings=generate_embedding,
            allow_dangerous_deserialization=True,
        )
    else:
        documents = load_data()
        embeddings = np.array([doc['embedding'] for doc in documents], dtype=np.float32)
        embeddings = np.vstack(embeddings)

        index = IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        faiss_index = FAISS(
            embedding_function=generate_embedding,
            index=index,
            docstore=InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)}),
            index_to_docstore_id={str(i): i for i in range(len(documents))},
        )
        faiss_index.save_local(folder_path)

        return faiss_index


def create_llm_chain():
    llm = ChatOpenAI(model='gpt-4', temperature=0.7, openai_api_key=OPENAI_API_KEY)
    return CHATGPT_SYSTEM_PROMPT | llm


if __name__ == '__main__':
    build_index()

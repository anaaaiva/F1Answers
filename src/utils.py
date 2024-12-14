import os

import numpy as np
import requests
from faiss import IndexFlatL2
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from settings import EMBEDDINGS_BASE_URL, EMBEDDINGS_HEADERS, EMBEDDINGS_MODEL


def generate_embeddings(input_text: str) -> list[float]:
    data = {'model': EMBEDDINGS_MODEL, 'input': input_text, 'dimensions': 1024}
    try:
        response = requests.post(
            EMBEDDINGS_BASE_URL, headers=EMBEDDINGS_HEADERS, json=data
        )
        response.raise_for_status()
        response = response.json().get('data', [])
        return np.array([doc['embedding'] for doc in response], dtype=np.float32)
    except requests.exceptions.RequestException as e:
        print('Error while requesting embedding:', e)
        return np.array([], dtype=np.float32)


def load_data():
    pdf_loader = PyPDFLoader(file_path='./data/regulations.pdf')
    pdf_documents = (
        pdf_loader.load() if os.path.exists('./data/regulations.pdf') else []
    )
    pdf_documents = [pdf_documents[0]]

    documents_with_embeddings = []
    for doc in pdf_documents:
        embedding = generate_embeddings(doc.page_content)
        if embedding.size > 0:
            documents_with_embeddings.append(
                {'text': doc.page_content, 'embedding': embedding}
            )

    return documents_with_embeddings


def build_index():
    index_path = './data'

    if os.path.exists(f'{index_path}/index.faiss'):
        return FAISS.load_local(index_path)
    else:
        documents = load_data()
        embeddings = np.array([doc['embedding'] for doc in documents], dtype=np.float32)
        embeddings = np.vstack(embeddings)

        index = IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        faiss_index = FAISS(
            embedding_function=generate_embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        faiss_index.save_local(index_path)

        return faiss_index

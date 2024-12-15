import os

import numpy as np
import requests
from faiss import IndexFlatL2, read_index, write_index
from langchain_openai import ChatOpenAI
from pypdf import PdfReader
from settings import (
    CHATGPT_MODEL,
    CHATGPT_SYSTEM_PROMPT,
    EMBEDDINGS_BASE_URL,
    EMBEDDINGS_HEADERS,
    EMBEDDINGS_MODEL,
    FAISS_INDEX_PATH,
    OPENAI_API_KEY,
    PDF_DIR_PATH,
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


def load_data(pdf_dir_path: str = PDF_DIR_PATH) -> list[dict]:
    documents = []

    for file in os.scandir(pdf_dir_path):
        if file.name.endswith('.pdf'):
            reader = PdfReader(file.path)

            for page in reader.pages:
                text = page.extract_text()

                if text.strip():
                    embedding = generate_embedding(text)

                    if embedding.size > 0:
                        documents.append({'text': text, 'embedding': embedding})

    return documents


def build_index(
    pdf_dir_path: str = PDF_DIR_PATH, faiss_index_path: str = FAISS_INDEX_PATH
) -> IndexFlatL2:
    if os.path.exists(faiss_index_path):
        return read_index(FAISS_INDEX_PATH)

    else:
        documents = load_data(pdf_dir_path)
        embeddings = np.array([doc['embedding'] for doc in documents], dtype=np.float32)

        index = IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        write_index(index, FAISS_INDEX_PATH)

        return index


def create_llm_chain():
    llm = ChatOpenAI(model=CHATGPT_MODEL, openai_api_key=OPENAI_API_KEY)
    return CHATGPT_SYSTEM_PROMPT | llm


if __name__ == '__main__':
    build_index()

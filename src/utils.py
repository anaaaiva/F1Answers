import os

import numpy as np
import requests
from faiss import IndexFlatL2, read_index, write_index
from pypdf import PdfReader
from settings import (
    API_HEADERS,
    CHATGPT_BASE_URL,
    CHATGPT_MODEL,
    CHATGPT_SYSTEM_PROMPT,
    EMBEDDINGS_BASE_URL,
    EMBEDDINGS_MODEL,
    FAISS_INDEX_PATH,
    PDF_DIR_PATH,
)


def generate_embedding(input_text: str) -> np.ndarray:
    data = {'model': EMBEDDINGS_MODEL, 'input': input_text, 'dimensions': 1024}

    try:
        response = requests.post(EMBEDDINGS_BASE_URL, headers=API_HEADERS, json=data)
        response.raise_for_status()
        embedding = response.json()['data'][0]['embedding']
        return np.array(embedding, dtype=np.float32)

    except requests.exceptions.RequestException as e:
        print(f'Error while generating embeddings: {e}')
        return np.array([], dtype=np.float32)


def load_data(pdf_dir_path: str = PDF_DIR_PATH) -> list[str]:
    documents = []

    for file in os.scandir(pdf_dir_path):
        if file.name.endswith('.pdf'):
            reader = PdfReader(file.path)

            for page in reader.pages:
                text = page.extract_text()

                if text.strip():
                    documents.append(text)

    return documents


def build_index(
    pdf_dir_path: str = PDF_DIR_PATH, faiss_index_path: str = FAISS_INDEX_PATH
) -> IndexFlatL2:
    if os.path.exists(faiss_index_path):
        return read_index(FAISS_INDEX_PATH)

    else:
        documents = load_data(pdf_dir_path)
        embeddings = [generate_embedding(text) for text in documents]
        embeddings = np.array(
            [embedding for embedding in embeddings if embedding.size], dtype=np.float32
        )

        index = IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        write_index(index, FAISS_INDEX_PATH)

        return index


def query_index(
    index: IndexFlatL2,
    query_embedding: np.ndarray,
    documents: list[str],
    top_k: int = 5,
) -> str:
    _, indices = index.search(query_embedding[np.newaxis, :], k=top_k)
    context = [documents[i] for i in indices[0] if i != -1]
    return ' '.join(context)


def generate_answer(context: str, question: str) -> str:
    data = {
        'model': CHATGPT_MODEL,
        'messages': [
            {'role': 'system', 'content': CHATGPT_SYSTEM_PROMPT},
            {'role': 'user', 'content': f'Context: {context}\n\nQuestion: {question}'},
        ],
        'temperature': 0.7,
    }
    try:
        response = requests.post(CHATGPT_BASE_URL, headers=API_HEADERS, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    except requests.exceptions.RequestException as e:
        print(f'Error while generating answer: {e}')
        return 'Failed to generate an answer.'


if __name__ == '__main__':
    build_index()

import os
from typing import Any, Dict, List

import numpy as np
import requests
from faiss import IndexFlatL2, read_index, write_index
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from pypdf import PdfReader
from settings import (
    API_HEADERS,
    EMBEDDER_BASE_URL,
    EMBEDDER_MODEL,
    FAISS_INDEX_PATH,
    GENERATOR_BASE_URL,
    GENERATOR_MODEL,
    GENERATOR_SYSTEM_PROMPT,
    PDF_DIR_PATH,
)


class CustomEmbeddings(Embeddings):
    # TODO: убрать неизменяемые блоки в инициализации (типо api_headers)
    def __init__(
        self,
        embedder_model: str = EMBEDDER_MODEL,
        embedder_base_url: str = EMBEDDER_BASE_URL,
        api_headers: Dict[str, str] = API_HEADERS,
    ):
        self.embedder_model = embedder_model
        self.embedder_base_url = embedder_base_url
        self.api_headers = api_headers

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        try:
            data = {"model": self.embedder_model, "input": text, "dimensions": 1024}
            response = requests.post(
                self.embedder_base_url, headers=self.api_headers, json=data
            )
            response.raise_for_status()
            embedding = response.json()["data"][0]["embedding"]
            return np.array(embedding, dtype=np.float32)

        except requests.exceptions.RequestException as e:
            print(f"Error while generating embeddings: {e}")
            return np.array([], dtype=np.float32)


class CustomLLM(LLM):
    # TODO: добавить внутренние переменные GENERATOR_MODEL и тд. или оставить так...
    def _call(self, prompt: str, **kwargs: Any) -> str:
        data = {
            "model": GENERATOR_MODEL,
            "messages": [
                {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
        }
        try:
            response = requests.post(GENERATOR_BASE_URL, headers=API_HEADERS, json=data)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            print(f"Error while generating answer: {e}")
            return "Failed to generate an answer."

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "custom"


def generate_embedding(input_text: str) -> np.ndarray:
    data = {"model": EMBEDDER_MODEL, "input": input_text, "dimensions": 1024}

    try:
        response = requests.post(EMBEDDER_BASE_URL, headers=API_HEADERS, json=data)
        response.raise_for_status()
        embedding = response.json()["data"][0]["embedding"]
        return np.array(embedding, dtype=np.float32)

    except requests.exceptions.RequestException as e:
        print(f"Error while generating embeddings: {e}")
        return np.array([], dtype=np.float32)


def load_data(pdf_dir_path: str = PDF_DIR_PATH) -> list[str]:
    documents = []

    for file in os.scandir(pdf_dir_path):
        if file.name.endswith(".pdf"):
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
            [embedding for embedding in embeddings if embedding.size > 0],
            dtype=np.float32,
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
    return " ".join(context)


def generate_answer(context: str, question: str) -> str:
    data = {
        "model": GENERATOR_MODEL,
        "messages": [
            {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
        ],
        "temperature": 0.7,
    }
    try:
        response = requests.post(GENERATOR_BASE_URL, headers=API_HEADERS, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        print(f"Error while generating answer: {e}")
        return "Failed to generate an answer."


if __name__ == "__main__":
    build_index()

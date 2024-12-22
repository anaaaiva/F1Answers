import numpy as np
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from settings import (
    API_HEADERS,
    EMBEDDER_BASE_URL,
    EMBEDDER_MODEL,
    GENERATOR_BASE_URL,
    GENERATOR_MODEL,
    GENERATOR_SYSTEM_PROMPT,
)


class CustomEmbeddings(Embeddings):
    # TODO: убрать неизменяемые блоки в инициализации (типо api_headers)
    def __init__(
        self,
        embedder_model: str = EMBEDDER_MODEL,
        embedder_base_url: str = EMBEDDER_BASE_URL,
        api_headers: dict[str, str] = API_HEADERS,
    ):
        self.embedder_model = embedder_model
        self.embedder_base_url = embedder_base_url
        self.api_headers = api_headers

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        try:
            data = {'model': self.embedder_model, 'input': text, 'dimensions': 1024}
            response = requests.post(
                self.embedder_base_url, headers=self.api_headers, json=data
            )
            response.raise_for_status()
            embedding = response.json()['data'][0]['embedding']
            return np.array(embedding, dtype=np.float32)

        except requests.exceptions.RequestException as e:
            print(f'Error while generating embeddings: {e}')
            return np.array([], dtype=np.float32)


class CustomLLM(LLM):
    # TODO: добавить внутренние переменные GENERATOR_MODEL и тд.
    def _call(self, prompt: str) -> str:
        data = {
            'model': GENERATOR_MODEL,
            'messages': [
                {'role': 'system', 'content': GENERATOR_SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt},
            ],
            'temperature': 0.7,
        }
        try:
            response = requests.post(GENERATOR_BASE_URL, headers=API_HEADERS, json=data)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']

        except requests.exceptions.RequestException as e:
            print(f'Error while generating answer: {e}')
            return 'Failed to generate an answer.'

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return 'custom'

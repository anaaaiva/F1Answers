import numpy as np
import requests
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from settings import (
    API_HEADERS,
    CONTEXTUALIZE_Q_SYSTEM_PROMPT,
    EMBEDDER_BASE_URL,
    EMBEDDER_MODEL,
    GENERATOR_BASE_URL,
    GENERATOR_MODEL,
    GENERATOR_SYSTEM_PROMPT,
    logger,
)


class CustomEmbeddings(Embeddings):
    """Custom embeddings class for document embedding using external API."""

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
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        try:
            data = {'model': self.embedder_model, 'input': text, 'dimensions': 1024}
            response = requests.post(
                self.embedder_base_url, headers=self.api_headers, json=data
            )
            response.raise_for_status()
            embedding = response.json()['data'][0]['embedding']
            return np.array(embedding, dtype=np.float32)

        except requests.exceptions.RequestException as e:
            logger.error(f'Error while generating embeddings: {e}')
            return np.array([], dtype=np.float32)


class CustomLLM(LLM):
    """Custom language model class for generating responses using external API."""

    def _call(self, prompt: str, **kwargs) -> str:
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
            logger.error(f'Error while generating answer: {e}')
            return 'Failed to generate an answer.'

    @property
    def _llm_type(self) -> str:
        """Return the type of language model used, for logging purposes only."""
        return 'custom'


def initialize_model(vectorstore: FAISS):
    """Initialize and return a retrieval-augmented generation (RAG) chain."""
    llm = CustomLLM()

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', CONTEXTUALIZE_Q_SYSTEM_PROMPT),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}'),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, vectorstore.as_retriever(), contextualize_q_prompt
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', GENERATOR_SYSTEM_PROMPT),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}'),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

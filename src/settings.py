import os

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate

load_dotenv()
OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY')

EMBEDDINGS_BASE_URL: str = os.getenv('EMBEDDINGS_BASE_URL')
EMBEDDINGS_MODEL: str = 'text-embedding-3-small'
EMBEDDINGS_HEADERS: dict = {
    'Authorization': f'Bearer {OPENAI_API_KEY}',
    'Content-Type': 'application/json',
}

PDF_DIR_PATH: str = './data/'
FAISS_INDEX_PATH: str = 'index.bin'

CHATGPT_BASE_URL: str = os.getenv('CHATGPT_BASE_URL')
CHATGPT_MODEL: str = 'gpt-4o-mini'
CHATGPT_SYSTEM_PROMPT = ChatPromptTemplate.from_template("""
    You are a Formula 1 assistant. Use the following context to answer the user's
    question as precisely as possible:\n\n{context}\n\nQuestion: {question}\n\nAnswer:
""")

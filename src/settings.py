import os

from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY')

EMBEDDINGS_BASE_URL: str = os.getenv('EMBEDDINGS_BASE_URL')
EMBEDDINGS_MODEL: str = 'text-embedding-3-small'
EMBEDDINGS_HEADERS: dict = {
    'Authorization': f'Bearer {OPENAI_API_KEY}',
    'Content-Type': 'application/json',
}

CHATGPT_BASE_URL: str = os.getenv('CHATGPT_BASE_URL')

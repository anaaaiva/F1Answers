import os

from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY')
API_HEADERS: dict = {
    'Authorization': f'Bearer {OPENAI_API_KEY}',
    'Content-Type': 'application/json',
}

EMBEDDER_BASE_URL: str = os.getenv('EMBEDDER_BASE_URL')
EMBEDDER_MODEL: str = 'text-embedding-3-small'

PDF_DIR_PATH: str = './data/'
FAISS_INDEX_PATH: str = './index/index.bin'

GENERATOR_BASE_URL: str = os.getenv('GENERATOR_BASE_URL')
GENERATOR_MODEL: str = 'gpt-4o-mini'
GENERATOR_SYSTEM_PROMPT = """
You are a Formula 1 assistant. Use the following context to answer the user's
question as precisely as possible:\n\n{context}\n\nQuestion: {question}\n\nAnswer:
"""

WIKI_SEARCH = [
    'Formula 1',
    'Formula 1 drivers',
    'Lewis Hamilton',
    'Max Verstappen',
    'Lando Norris',
]

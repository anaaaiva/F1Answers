import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_texts(directory: str) -> list:
    texts = []
    for file in os.listdir(directory):
        if file.endswith('.txt'):
            with open(os.path.join(directory, file), 'r', encoding='utf8') as f:
                texts.append((file, f.read()))
    return texts


def create_embeddings(texts: list, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = {}
    for title, text in texts:
        embeddings[title] = model.encode(text, convert_to_tensor=True)
    return embeddings


def build_faiss_index(embeddings):
    dimension = embeddings[next(iter(embeddings))].shape[0]
    index = faiss.IndexFlatL2(dimension)
    titles = list(embeddings.keys())
    matrix = np.array([embeddings[title].numpy() for title in titles])
    index.add(matrix)
    return index, titles


def search(query_embedding, index, titles, top_k=5) -> list:
    distances, indices = index.search(np.array([query_embedding.numpy()]), top_k)
    return [(titles[idx], distances[0][i]) for i, idx in enumerate(indices[0])]


if __name__ == '__main__':
    wiki_texts = load_texts('./data/')
    embeddings = create_embeddings(wiki_texts)

    index, titles = build_faiss_index(embeddings)

    query = 'Who won the 2020 Formula 1 World Championship?'
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query, convert_to_tensor=True)
    results = search(query_embedding, index, titles)
    print('Search results:', results)

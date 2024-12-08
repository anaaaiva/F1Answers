import streamlit as st
from sentence_transformers import SentenceTransformer
from utils import build_faiss_index, create_embeddings, load_texts, search


def main():
    st.title('Formula 1 RAG System')

    model = SentenceTransformer('all-MiniLM-L6-v2')
    wiki_texts = load_texts('./data/')
    embeddings = create_embeddings(wiki_texts)
    index, titles = build_faiss_index(embeddings)

    query = st.text_input('Enter your question about Formula 1:')
    if query:
        query_embedding = model.encode(query, convert_to_tensor=True)
        results = search(query_embedding, index, titles)

        st.write(f'Search results for: {query}')
        for title, dist in results:
            st.write(f'Title: {title}, Distance: {dist}')


if __name__ == '__main__':
    main()

import streamlit as st
from utils import build_index, generate_embeddings


def main():
    st.title('Formula 1 RAG System')

    faiss_index = build_index()
    retriever = faiss_index.as_retriever()

    query = st.text_input('Enter your question about Formula 1:')

    if query:
        query_embedding = generate_embeddings(query)

        if query_embedding.size > 0:
            with st.spinner('Fetching the answer...'):
                results = retriever.get_relevant_documents(query_embedding.tolist())
                st.success('Top results:')
                for result in results:
                    st.write(result.text)
        else:
            st.error('Failed to generate embedding for the query.')


if __name__ == '__main__':
    main()

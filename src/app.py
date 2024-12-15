import streamlit as st
from utils import (
    build_index,
    generate_answer,
    generate_embedding,
    load_data,
    query_index,
)


def main():
    st.title('F1Answers Bot')

    index = build_index()
    documents = load_data()

    query = st.text_input('Enter your question about Formula 1:')

    if query:
        query_embedding = generate_embedding(query)

        if query_embedding.size > 0:
            with st.spinner('Fetching relevant context...'):
                context = query_index(index, query_embedding, documents)

            with st.spinner('Generating answer...'):
                answer = generate_answer(context, query)

            st.write(answer)
        else:
            st.error('Failed to generate embedding for the query.')


if __name__ == '__main__':
    main()

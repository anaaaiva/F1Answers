import streamlit as st
from langchain.schema import HumanMessage
from utils import build_index, create_llm_chain, generate_embeddings


def main():
    st.title('F1Answers Bot')

    faiss_index = build_index()
    retriever = faiss_index.as_retriever()

    query = st.text_input('Enter your question about Formula 1:')

    if query:
        query_embedding = generate_embeddings(query)

        if query_embedding.size > 0:
            with st.spinner('Fetching relevant context...'):
                results = retriever.get_relevant_documents(query_embedding.tolist())
                context = ' '.join([result.text for result in results])

            if context:
                llm_chain = create_llm_chain()
                messages = [
                    HumanMessage(content=f'Context: {context}\n\nQuestion: {query}')
                ]
                with st.spinner('Generating answer...'):
                    answer = llm_chain(messages).content
                st.success('Generated Answer:')
                st.write(answer)
            else:
                st.error('No relevant context found for your question.')
        else:
            st.error('Failed to generate embedding for the query.')


if __name__ == '__main__':
    main()

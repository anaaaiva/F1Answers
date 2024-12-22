import streamlit as st
from data_loading import load_data
from langchain_core.messages import HumanMessage
from model_initialisation import initialize_model
from processing import process_data


def format_source(source):
    """Format the source information for better readability."""
    metadata = source.metadata
    title = metadata.get('title', 'Unknown Title')
    url = metadata.get('source', 'Unknown Source')
    return f'**Title**: {title}\n**Sourse**: {url}\n'


def display_chat_history(chat_history: list):
    """Display the chat history."""
    if not chat_history:
        st.warning('No chat history to display.')
        return

    st.subheader('Chat History')
    for i, msg in enumerate(chat_history):
        if isinstance(msg, HumanMessage):
            st.markdown(f'**Query {i//2 + 1}:** {msg.content}')
        else:
            st.markdown(f'**Answer {i//2 + 1}:** {msg}')


def main():
    st.title('F1Answers Bot')

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if st.sidebar.button('Clear Chat History'):
        st.session_state.chat_history = []
        st.sidebar.success('Chat history cleared.')

    if 'docs_all' not in st.session_state:
        st.session_state.docs_all = load_data()

    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = process_data(st.session_state.docs_all)

    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = initialize_model(st.session_state.vectorstore)

    query = st.text_input('Enter your question about Formula 1:')
    if st.button('Ask'):
        if query:
            result = st.session_state.rag_chain.invoke(
                {'input': query, 'chat_history': st.session_state.chat_history}
            )
            answer = result['answer']
            context = result['context']

            st.markdown(f'**Answer:** {answer}')

            st.markdown('**Source(s):**')
            displayed_urls = set()
            for doc in context:
                metadata = doc.metadata
                url = metadata.get('source', 'Unknown Source')
                if url not in displayed_urls:
                    st.markdown(format_source(doc))
                    displayed_urls.add(url)

            st.session_state.chat_history.extend([HumanMessage(content=query), answer])

    if st.sidebar.button('Display Chat History'):
        display_chat_history(st.session_state.chat_history)

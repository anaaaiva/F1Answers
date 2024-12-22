import streamlit as st
from data_loading import load_data, prepare_data
from langchain_core.messages import HumanMessage
from model_initialisation import initialize_model


def format_source(source) -> str:
    """Format the source information for better readability."""
    metadata = source.metadata
    title = metadata.get('title', 'Unknown Title')
    url = metadata.get('source', 'Unknown Source')
    return f'**Title**: {title}\n**Sourse**: {url}\n'


def display_chat_history(chat_history: list) -> None:
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

    if st.sidebar.button('Clear chat history'):
        st.session_state.chat_history = []
        st.sidebar.success('Chat history cleared')

    if 'documents' not in st.session_state:
        st.session_state.documents = load_data()

    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = prepare_data(st.session_state.documents)

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

            for document in context:
                url = document.metadata.get('source', 'Unknown source')

                if url not in displayed_urls:
                    st.markdown(format_source(document))
                    displayed_urls.add(url)

            st.session_state.chat_history.extend([HumanMessage(content=query), answer])

    if st.sidebar.button('Display chat history'):
        display_chat_history(st.session_state.chat_history)

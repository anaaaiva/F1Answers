import streamlit as st
from data_loading import format_source, prepare_data
from model_initialisation import initialize_model


def main():
    st.title('F1Answers Bot')

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    rag_chain = initialize_model(prepare_data())

    if user_input := st.chat_input('Ask me anything about Formula 1:'):
        with st.chat_message('user'):
            st.markdown(user_input)

        with st.chat_message('assistant'):
            result = rag_chain.invoke(
                {'input': user_input, 'chat_history': st.session_state.messages}
            )
            st.session_state.messages.append({'role': 'user', 'content': user_input})
            answer, context = result['answer'], result['context']

            displayed_urls = set()
            sources_text = '\n\n**Sources:**\n\n'

            for document in context:
                url = document.metadata.get('source', 'Unknown source')

                if url not in displayed_urls:
                    sources_text += format_source(document) + '\n'
                    displayed_urls.add(url)

            response = answer + sources_text
            st.markdown(response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})


if __name__ == '__main__':
    main()

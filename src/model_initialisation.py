from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from settings import CONTEXTUALIZE_Q_SYSTEM_PROMPT, GENERATOR_SYSTEM_PROMPT
from utils import CustomLLM


def initialize_model(vectorstore: FAISS):
    """Initialize and return a retrieval-augmented generation (RAG) chain."""
    llm = CustomLLM()

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', CONTEXTUALIZE_Q_SYSTEM_PROMPT),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}'),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, vectorstore.as_retriever(), contextualize_q_prompt
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', GENERATOR_SYSTEM_PROMPT),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}'),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

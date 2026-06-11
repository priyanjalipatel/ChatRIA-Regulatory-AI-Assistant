import streamlit as st
import os

from dotenv import load_dotenv


from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings
)

from langchain_community.vectorstores import FAISS


from langchain_core.prompts import PromptTemplate


from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)



# =========================
# ENV
# =========================

load_dotenv()



VECTOR_STORE_PATH = "usfda_vectorstore"



# =========================
# LOAD VECTOR STORE
# =========================


@st.cache_resource
def load_vectorstore():


    embeddings = OpenAIEmbeddings()


    return FAISS.load_local(

        VECTOR_STORE_PATH,

        embeddings,

        allow_dangerous_deserialization=True

    )



vectorstore = load_vectorstore()



# =========================
# BUILD RAG CHAIN
# =========================


def build_qa_chain():


    retriever = vectorstore.as_retriever(

        search_type="mmr",

        search_kwargs={

            "k":10,

            "fetch_k":50

        }

    )



    template = """

You are ChatRIA, a pharmaceutical regulatory assistant.

Answer using FDA document context.

Provide detailed scientific answers.

End with an executive summary.


Context:

{context}


Question:

{input}


Answer:

"""


    prompt = PromptTemplate.from_template(
        template
    )



    llm = ChatOpenAI(

        model="gpt-4o",

        temperature=0

    )



    document_chain = create_stuff_documents_chain(

        llm,

        prompt

    )



    rag_chain = create_retrieval_chain(

        retriever,

        document_chain

    )


    return rag_chain




qa_chain = build_qa_chain()



# =========================
# STREAMLIT UI
# =========================


st.set_page_config(

    page_title="USFDA Document QA",

    layout="wide"

)



st.title(
    "USFDA Document Question-Answering System"
)



st.markdown(
    "Ask questions across FDA regulatory documents."
)



with st.expander(
    "Example Questions"
):

    st.write(
"""
- Compare mechanism of action between OPDIVO and YERVOY.
- Common adverse reactions shared by TECENTRIQ and PROLEUKIN.
- Compare dosing recommendations of BRAFTOVI and MEKTOVI.
"""
    )



question = st.text_input(
    "Enter your question:"
)



if question:


    with st.spinner(
        "Generating answer..."
    ):


        result = qa_chain.invoke(

            {

                "input":question

            }

        )



    st.subheader(
        "Answer"
    )


    st.write(
        result["answer"]
    )



    st.subheader(
        "Sources"
    )


    for i,doc in enumerate(
        result["context"],
        1
    ):


        st.markdown(

f"""
**{i}.**

File:

{doc.metadata.get("source_file")}

Page:

{doc.metadata.get("page")}

"""
        )

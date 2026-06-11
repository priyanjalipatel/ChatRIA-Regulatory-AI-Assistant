import os
import glob
import re
import logging

from typing import List

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI

from langchain_core.prompts import PromptTemplate

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# =========================
# Logging
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# =========================
# ENV
# =========================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# =========================
# DOCUMENT PROCESSOR
# =========================


class USFDADocumentProcessor:


    def __init__(
        self,
        pdf_folder="USFDA",
        vector_store_path="usfda_vectorstore"
    ):

        self.pdf_folder = pdf_folder
        self.vector_store_path = vector_store_path
        self.vector_store = None



    # ---------------------

    def get_pdf_files(self):

        files = glob.glob(
            os.path.join(self.pdf_folder,"*.pdf")
        )

        logger.info(
            f"Found {len(files)} PDFs"
        )

        return files



    # ---------------------

    def preprocess_text(self,text):

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        text = re.sub(
            r"\[\d+\]",
            "",
            text
        )

        text = re.sub(
            r"(\w+)-\s+(\w+)",
            r"\1\2",
            text
        )


        return text.strip()



    # ---------------------

    def load_pdfs(self):

        documents=[]


        for pdf in self.get_pdf_files():

            logger.info(
                f"Loading {pdf}"
            )


            loader = PyPDFLoader(pdf)


            docs = loader.load()



            for d in docs:

                d.metadata["source_file"] = os.path.basename(pdf)

                d.page_content = self.preprocess_text(
                    d.page_content
                )


            documents.extend(docs)



        logger.info(
            f"Loaded {len(documents)} pages"
        )


        return documents




    # ---------------------

    def split_documents(self,documents):

        splitter = RecursiveCharacterTextSplitter(

            chunk_size=500,

            chunk_overlap=150

        )


        chunks = splitter.split_documents(
            documents
        )


        logger.info(
            f"Created {len(chunks)} chunks"
        )


        return chunks



    # ---------------------

    def create_vector_store(self):


        if os.path.exists(self.vector_store_path):

            logger.info(
                "Loading existing FAISS"
            )


            self.vector_store = FAISS.load_local(

                self.vector_store_path,

                OpenAIEmbeddings(),

                allow_dangerous_deserialization=True

            )


            return self.vector_store



        docs = self.load_pdfs()


        chunks = self.split_documents(
            docs
        )



        self.vector_store = FAISS.from_documents(

            chunks,

            OpenAIEmbeddings()

        )


        self.vector_store.save_local(

            self.vector_store_path

        )


        return self.vector_store



    # =========================
    # RAG CHAIN MODERN VERSION
    # =========================


    def create_qa_chain(self):


        if self.vector_store is None:

            self.create_vector_store()



        retriever = self.vector_store.as_retriever(

            search_type="mmr",

            search_kwargs={

                "k":10,

                "fetch_k":50

            }

        )



        prompt = PromptTemplate.from_template(

"""
You are ChatRIA, a pharmaceutical regulatory assistant.

Answer using only FDA document context.

Provide:
- detailed explanation
- structured sections
- executive summary

Context:

{context}


Question:

{input}


Answer:
"""

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



    # ---------------------


    def answer_question(self,question):


        chain = self.create_qa_chain()


        response = chain.invoke(

            {

                "input":question

            }

        )


        answer=response["answer"]



        sources=[]


        for doc in response["context"]:


            sources.append(

                {

                "file":
                doc.metadata.get(
                    "source_file"
                ),

                "page":
                doc.metadata.get(
                    "page"
                )

                }

            )



        return {


            "answer":answer,

            "sources":sources

        }




# =========================
# MAIN
# =========================


def main():


    processor = USFDADocumentProcessor()



    processor.create_vector_store()



    print(
        "\nUSFDA RAG Assistant"
    )

    print(
        "Type exit to quit"
    )



    while True:


        question=input(
            "\nQuestion: "
        )



        if question.lower()=="exit":

            break



        result = processor.answer_question(

            question

        )



        print(
            "\nANSWER:\n"
        )


        print(
            result["answer"]
        )



        print(
            "\nSOURCES:"
        )


        for s in result["sources"]:

            print(s)




if __name__=="__main__":

    main()

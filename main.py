import os
import fitz
from pathlib import Path
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec

from langchain_community.vectorstores import Pinecone as PC
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from g4f.client import Client

from sentence_transformers import SentenceTransformer
EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

global_instance = None
g4f_client = Client()
load_dotenv()

PDF_STORAGE_PATH = os.environ.get("PDF_STORAGE_PATH")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
PINECONE_INSTANCE = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))


class TacticalEdgeAssistant:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings()
        self.pinecone_db = self.get_db_instance()

    @staticmethod
    def store_new_embedding(pdf_content, file_name):
        # Convert the text into the Document format required by langchain
        document = Document(
            page_content=pdf_content,
            metadata={"source": file_name}
        )
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        split_documents = text_splitter.split_documents([document])

        index = PINECONE_INSTANCE.Index(PINECONE_INDEX)

        for i, chunk in enumerate(split_documents):
            # Generate embeddings for the chunk using HuggingFace
            embedding = EMBEDDING_MODEL.encode(chunk.page_content).tolist()

            # Store the embedding in Pinecone
            index.upsert([(f'doc-{i}', embedding)])

    @staticmethod
    def load_pdf(file_path):
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()

        # Convert the text into the Document format required by langchain
        document = Document(
            page_content=text,
            metadata={"source": file_path}
        )
        return [document]

    def load_documents(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        split_documents = []

        # doc_to_insert = "/Users/mac/Downloads/ZIGUP-FY24-Results-Deck.pdf"
        # documents = self.load_pdf(doc_to_insert)
        # for document in documents:
        #     split_documents.extend(text_splitter.split_documents([document]))

        pdf_directory = Path(PDF_STORAGE_PATH)
        for file_name in os.listdir(pdf_directory):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(pdf_directory, file_name)
                documents = self.load_pdf(file_path)
                for document in documents:
                    split_documents.extend(text_splitter.split_documents([document]))

        return split_documents

    def get_db_instance(self):
        pine_indexes = PINECONE_INSTANCE.list_indexes().names()
        if PINECONE_INDEX not in pine_indexes:
            docs = self.load_documents()
            PINECONE_INSTANCE.create_index(
                name=PINECONE_INDEX,
                dimension=768,
                metric='euclidean',
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            pinecone_db = PC.from_documents(docs, self.embeddings, index_name=PINECONE_INDEX)
        else:
            pinecone_db = PC.from_existing_index(PINECONE_INDEX, self.embeddings)

        return pinecone_db

    @staticmethod
    def augmentation(context, user_input):
        prompt = f"""
            You are an AI assistant with access to a detailed financial and business performance report for FY2024.
            Use the context provided to answer questions accurately and comprehensively.

            Context: {context}
            Your task is to provide a concise and clear response based on this context.
            If the context doesn't contain enough information to answer the question,
            please indicate that the information is not available.

            Question: {user_input}
            Answer: """

        response = g4f_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        reply = response.choices[0].message.content
        print(f"Response: {reply}")
        return reply

    def answer_retriever(self, user_input):
        retriever = self.pinecone_db.as_retriever(search_type="mmr", search_kwargs={"k": 3})
        matched_docs = retriever.invoke(user_input)
        context = matched_docs[0]
        source = (context.metadata["source"]).split("/")[-1].replace(".pdf", "").strip()
        return matched_docs[0].page_content, source


def get_ai_search_instance():
    global global_instance
    if global_instance is None:
        global_instance = TacticalEdgeAssistant()
    return global_instance


def main():
    try:
        # user_input = "What was UK&I Rental profit in FY2023?"
        user_input = "list of best animated movies in 2024?"
        ai_search = get_ai_search_instance()

        answer, source = ai_search.answer_retriever(user_input)
        refined_answer = ai_search.augmentation(answer, user_input)

        print("===================")
        print(refined_answer)
        print("===================")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()

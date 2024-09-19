import os
import fitz
import openai
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec

from langchain_community.vectorstores import Pinecone as PC
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

global_instance = None
load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
PINECONE_INSTANCE = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))


class TacticalEdgeAssistant:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings()
        self.pinecone_index = self.get_db_instance()
        self.pinecone_db = None
        self.retriever = self.initialize_retriever()

    def initialize_retriever(self):
        pinecone_db = PC.from_existing_index(PINECONE_INDEX, self.embeddings)
        retriever = pinecone_db.as_retriever(search_type="mmr", search_kwargs={"k": 3})
        return retriever

    def store_new_embedding(self, pdf_content, file_name):
        # Convert the text into the Document format required by langchain
        document = Document(
            page_content=pdf_content,
            metadata={"source": file_name}
        )
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        split_documents = text_splitter.split_documents([document])

        for i, chunk in enumerate(split_documents):
            # Generate embeddings for the chunk using HuggingFace
            embedding = EMBEDDING_MODEL.encode(chunk.page_content).tolist()

            # Store the embedding in Pinecone
            self.pinecone_index.upsert([(f'{file_name}-doc-{i}', embedding, {"source": file_name, "text": chunk.page_content})])

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

    def get_db_instance(self):
        pine_indexes = PINECONE_INSTANCE.list_indexes().names()
        if PINECONE_INDEX not in pine_indexes:
            PINECONE_INSTANCE.create_index(
                name=PINECONE_INDEX,
                dimension=768,
                metric='euclidean',
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        return PINECONE_INSTANCE.Index(PINECONE_INDEX)

    @staticmethod
    def augmentation(context, user_input):
        try:
            prompt = f"""
                You are an AI assistant with access to a detailed financial and business performance report for FY2024.
                Use the context provided to answer questions accurately and comprehensively.

                Context: {context}
                Your task is to provide a concise and clear response based on this context.
                If the context doesn't contain enough information to answer the question,
                please indicate that the information is not available.

                Question: {user_input}
                Answer: """

            response = openai.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

            reply = response.choices[0].message.content
            print(f"Response: {reply}")
            return reply
        except Exception as ex:
            print(ex)
            raise Exception("Error: Rate limit exceeded, please try again")

    def answer_retriever(self, user_input):
        context_content, source = None, None
        if not self.retriever:
            return None, None

        matched_docs = self.retriever.invoke(user_input)
        if matched_docs:
            context = matched_docs[0]
            context_content = matched_docs[0].page_content
            source = (context.metadata["source"]).split("/")[-1].replace(".pdf", "").strip()

        return context_content, source


def get_ai_search_instance():
    global global_instance
    if global_instance is None:
        global_instance = TacticalEdgeAssistant()
    return global_instance


def main():
    try:
        user_input = "What was UK&I Rental profit in FY2023?"
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

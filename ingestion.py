import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    loader = TextLoader(
        "C:\\Users\\romer\\repos\\langchain-course\\mediumblog1.txt", 
        encoding="utf-8")
    document = loader.load()

    print("splitting...")

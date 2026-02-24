import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

EMBEDDING_DIMENSION = 768  # nomic-embed-text output dimension

def recreate_pinecone_index(pc: Pinecone, index_name: str, dimension: int) -> None:
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name in existing:
        print(f">> Deleting existing index '{index_name}'...")
        pc.delete_index(index_name)

    print(f">> Creating index '{index_name}' with dimension {dimension}...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(">> Index created.")


if __name__ == '__main__':
    print("Loading text...")
    loader = TextLoader(
        "C:\\Users\\romer\\repos\\langchain-course\\mediumblog1.txt",
        encoding="utf-8",
    )
    document = loader.load()

    print(">> Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f">>> Number of chunks: {len(texts)}")

    index_name = os.environ.get("PINECONE_INDEX_NAME")
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    recreate_pinecone_index(pc, index_name, EMBEDDING_DIMENSION)

    embeddings = OllamaEmbeddings(model=os.environ.get("OLLAMA_MODEL", "nomic-embed-text"))
    print(">> Ingesting into Pinecone...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=index_name)
    print(">> Done!")

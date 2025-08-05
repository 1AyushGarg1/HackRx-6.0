import os
import asyncio
import time
from dotenv import load_dotenv

# Pinecone
from pinecone import Pinecone, ServerlessSpec

# LlamaIndex
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    Document
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.readers.remote import RemoteReader

# === 1. Load Environment Variables ===
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# === 2. Configure Global LlamaIndex Settings ===
# This is the modern, recommended way to configure the LLM and embedding model.
Settings.llm = Groq(model="llama3-70b-8192", api_key=groq_api_key)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

async def wait_for_pinecone_indexing(pinecone_index, expected_vectors: int, timeout: int = 180):
    """
    Asynchronously polls the Pinecone index until the number of vectors
    matches the expected count or a timeout is reached.
    """
    print(f"⏳ Waiting for {expected_vectors} vectors to be indexed in Pinecone...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            stats = pinecone_index.describe_index_stats()
            if stats.total_vector_count >= expected_vectors:
                print(f"✅ Indexing complete! Found {stats.total_vector_count} vectors.")
                return True
            # print(f"   Current vector count: {stats.total_vector_count}/{expected_vectors}...")
            # Use asyncio.sleep for a non-blocking wait in an async function
            await asyncio.sleep(5)
        except Exception as e:
            print(f"An error occurred while checking index stats: {e}")
            await asyncio.sleep(5)
    
    raise TimeoutError(f"Pinecone indexing timed out after {timeout} seconds.")

async def build_or_load_index(url: str, index_name: str, pc: Pinecone):
    """
    Creates a new Pinecone index and populates it if it doesn't exist or is empty.
    If it already exists and is populated, it loads the index directly.
    """
    # Check if the index exists. If not, create it.
    if index_name not in [idx['name'] for idx in pc.list_indexes().indexes]:
        # print(f"Index '{index_name}' not found. Creating a new one...")
        pc.create_index(
            name=index_name,
            dimension=768,  # Must match your embedding model's dimension (all-mpnet-base-v2 is 768)
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        # Wait a moment for the index to be ready after creation
        await asyncio.sleep(5)

    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    # Check if the index is empty. If so, populate it.
    stats = pinecone_index.describe_index_stats()
    if stats.total_vector_count == 0:
        print(f"Index '{index_name}' is empty. Populating from URL...")
        
        # Load data from the remote URL
        # print(f"🔄 Loading PDF document from: {url}")
        loader = RemoteReader()
        documents = loader.load_data(url=url)
        # print(f"📄 Loaded {len(documents)} document(s).")

        # Manually parse nodes to get an exact count for polling
        node_parser = Settings.node_parser
        nodes = node_parser.get_nodes_from_documents(documents)
        expected_nodes = len(nodes)
        
        # Create the storage context and populate the index
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # print(f"📦 Populating index with {expected_nodes} document nodes...")
        index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
        
        # Use the intelligent polling function to wait for indexing to complete
        await wait_for_pinecone_indexing(pinecone_index, expected_nodes)
    else:
        # print(f"✅ Index '{index_name}' already contains {stats.total_vector_count} vectors. Loading index directly.")
        # If already populated, just load it from the existing vector store
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    return index

async def answer_query(query: str, query_engine):
    """Asynchronously queries the engine and prints the response."""
    # print(f"\n💬 Query: {query}")
    response = await query_engine.aquery(query)
    
    # print("\n🧠 Response:")
    # print("=" * 60)
    # print(response.response.strip() or "⚠️ No relevant text found in the document for this query.")
    return (response.response.strip() or "⚠️ No relevant text found in the document for this query.")

async def main():
    """
    Main asynchronous function to set up the RAG pipeline and run queries.
    """
    url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    index_name = "ayush-trail-instance"
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=pinecone_api_key)

    # Build the index if it's new/empty, or load it if it exists
    vector_index = await build_or_load_index(url, index_name, pc)

    # Create the query engine
    query_engine = vector_index.as_query_engine(similarity_top_k=4)
    print("\n✅ Query engine is ready.")

    # Run queries
    reponse = await answer_query("does the knee surgery included in this policy?", query_engine)
    respones = await answer_query("What is the policy name and what does it cover?", query_engine)
    print(respones)
    print(reponse)


if __name__ == "__main__":
    asyncio.run(main())
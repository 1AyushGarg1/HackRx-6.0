import os
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
# from llama_index.core.response.pprint_utils import pprint_response
from llama_index.readers.remote import RemoteReader
loader = RemoteReader()

llm = Groq(model="llama3-70b-8192", api_key=groq_api_key)

# retriever =  VectorIndexRetriever(index=index,similarity_top_k=4)
# postProcessor = SimilarityPostprocessor(similarity_cutoff=0.6)
# query_engine = None

def pdfToEmbeddings(url):
    documents = loader.load_data(url=url)
    # print(".....................................................")
    # print("documents are: ",documents)
    # print("..........................................................")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    index = VectorStoreIndex.from_documents(documents=documents,show_progress=True)
    retriever =  VectorIndexRetriever(index=index,similarity_top_k=4)
    postProcessor = SimilarityPostprocessor(similarity_cutoff=0.6)
    query_engine = RetrieverQueryEngine.from_args(retriever=retriever,node_postprocessors=[postProcessor] ,llm=llm)
    return query_engine


def answerAQuery(query:str,query_engine):
    response = query_engine.query(query)
    return response.response


async def main():
    query_engine = pdfToEmbeddings("https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D")
    response = answerAQuery("what is the CIN no in this policy?",query_engine)
    print(response)
    return

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

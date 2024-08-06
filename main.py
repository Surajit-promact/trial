# pip install python-dotenv, llama-index, llama-index-embeddings-huggingface, 
# chromadb, llama-index-vector-stores-chroma, llama-index-llms-gemini, google-generativeai

import os
import chromadb
from dotenv import load_dotenv

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.llms import ChatMessage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.gemini import Gemini



#1 load env variables, keys, credentials etc
load_dotenv()
#openai_api = os.getenv("OPENAI_API_KEY")
google_api = os.getenv("GOOGLE_API_KEY")


#2 load/read docs
documents = SimpleDirectoryReader("./data").load_data()
#print(documents)
print("Documents loading done...")


#3 embed docs
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model
Settings.chunk_size = 512

#index = VectorStoreIndex.from_documents(documents)
#print(index)



#4 save the indexed docs
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("trial")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
print(index)
print("Documents indexing and saving done...")


#5 query engine and query
llm = Gemini(model="models/gemini-1.5-pro")
Settings.llm = llm

query_engine = index.as_query_engine()
response = query_engine.query("What is minimum notice period in the company?")
print(response)


#6 chatbot
print("----------------------")
messages = []
start=input('to start/stop chat, enter y/n')
while start=='y':
    user_question = input('enter question: ')
    messages.append(ChatMessage(role="user", content=user_question))

    resp = llm.chat(messages)
    messages.append(ChatMessage(role="assistant", content=resp.message.content))
    print(messages)

    start=input('to continue/stop chat, enter y/n')

print("----------------------")


#7 agent and chatbot



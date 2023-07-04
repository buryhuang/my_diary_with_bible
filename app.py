import json
import os
from os import getenv
from llama_index import SimpleDirectoryReader, ServiceContext, LLMPredictor
from llama_index import VectorStoreIndex, StorageContext
from llama_index.vector_stores import OpensearchVectorStore, OpensearchVectorClient
from langchain.chat_models import ChatOpenAI

from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes


@app.route('/load', methods=['POST'])
def load():
    endpoint = getenv("OPENSEARCH_ENDPOINT", "http://localhost:9200")
    idx = getenv("OPENSEARCH_INDEX", "buryhuang-gpt-index-demo")
    client = OpensearchVectorClient(endpoint, idx, 1536, embedding_field="embedding", text_field="content")
    vector_store = OpensearchVectorStore(client)
    for file in os.listdir('llmindex_docs/output'):
        print(f"Indexing {file}")
        documents = SimpleDirectoryReader(input_files=[f'llmindex_docs/output/{file}'],
                                          filename_as_id=True).load_data()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(documents=documents, storage_context=storage_context)
    return "Indexing completed", 200


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_id = data['user_id']
    query = data['query']

    # Load chat history for user_id
    chat_history = load_context(user_id)

    # Process the query
    response = process_query(query, chat_history)

    # Save chat history for user_id
    chat_history.append((query, response))
    persist_context(user_id, chat_history)

    return { 'response': response }, 200


def persist_context(user_id, chat_history):
    chat_history_dict = []
    for (human_msg, agent_message) in chat_history:
        chat_history_dict.append({
            "human_message": human_msg,
            "agent_message": agent_message
        })
    file_name = f"chat_history_{user_id}.json"
    with open(file_name, "w") as f:
        f.write(json.dumps(chat_history_dict))


def load_context(user_id):
    file_name = f"chat_history_{user_id}.json"
    try:
        with open(file_name, "r") as f:
            chat_history_dict = json.loads(f.read())
        chat_history = []
        for item in chat_history_dict:
            chat_history.append((item["human_message"], item["agent_message"]))
    except:
        chat_history = []
    return chat_history


def process_query(query, chat_history):
    endpoint = getenv("OPENSEARCH_ENDPOINT", "http://localhost:9200")
    idx = getenv("OPENSEARCH_INDEX", "buryhuang-gpt-index-demo")
    client = OpensearchVectorClient(endpoint, idx, 1536, embedding_field="embedding", text_field="content")
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size=512)

    vector_store = OpensearchVectorStore(client)
    index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)

    chat_history_in_prompt = ""
    for (human_msg, agent_message) in chat_history:
        chat_history_in_prompt += f"Human: {human_msg}\nAssistant: {agent_message}\n"

    custom_prompt = f"""
        Given a conversation (between Human and Assistant) and a follow up message from Human,
        continue the conversation that captures all relevant context from the chat history. 

        <Chat History> 
        {chat_history_in_prompt}

        <Follow Up Message>
        Human: {query}
    """

    query_engine = index.as_query_engine(service_context=service_context, similarity_top_k=5, streaming=False)
    res = query_engine.query(custom_prompt)
    return res.response


if __name__ == '__main__':
    app.run()

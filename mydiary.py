import argparse
import json
import os
import sys
from os import getenv

import boto3
from botocore.exceptions import NoCredentialsError
from langchain.chat_models import ChatOpenAI
from llama_index import SimpleDirectoryReader, ServiceContext, LLMPredictor
from llama_index import VectorStoreIndex, StorageContext
from llama_index.vector_stores import OpensearchVectorStore, OpensearchVectorClient


def upload_to_s3(local_file_path, bucket_name):
    s3 = boto3.client('s3')
    s3_object_name = "bury/" + os.path.basename(local_file_path)

    try:
        s3.upload_file(local_file_path, bucket_name, s3_object_name)
        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_object_name}"
        return s3_url
    except FileNotFoundError:
        print("The file was not found")
        return None
    except NoCredentialsError:
        print("Credentials not available")
        return None


def persist_context(chat_history):
    chat_history_dict = []
    for (human_msg, agent_message) in chat_history:
        chat_history_dict.append({
            "human_message": human_msg,
            "agent_message": agent_message
        })
    with open("chat_history.json", "w") as f:
        f.write(json.dumps(chat_history_dict))


def load_context():
    try:
        with open("chat_history.json", "r") as f:
            chat_history_dict = json.loads(f.read())
        chat_history = []
        for item in chat_history_dict:
            chat_history.append((item["human_message"], item["agent_message"]))
    except:
        chat_history = []
    return chat_history


def main(args):
    if args.load:
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
        sys.exit(0)

    if args.query:
        endpoint = getenv("OPENSEARCH_ENDPOINT", "http://localhost:9200")
        idx = getenv("OPENSEARCH_INDEX", "buryhuang-gpt-index-demo")
        client = OpensearchVectorClient(endpoint, idx, 1536, embedding_field="embedding", text_field="content")
        llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True))
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size=512)

        vector_store = OpensearchVectorStore(client)
        index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)
        custom_chat_history = load_context()
        try:
            while True:
                query_str = input("Enter your question: ")
                query_engine = index.as_query_engine()

                chat_history_in_prompt = ""
                for (human_msg, agent_message) in custom_chat_history:
                    chat_history_in_prompt += f"Human: {human_msg}\nAssistant: {agent_message}\n"

                custom_prompt = f"""
                    Given a conversation (between Human and Assistant) and a follow up message from Human,
                    continue the conversation that captures all relevant context from the chat history. 
    
                    <Chat History> 
                    {chat_history_in_prompt}
    
                    <Follow Up Message>
                    Human: {query_str}
                """

                # print(custom_prompt)

                res = query_engine.query(custom_prompt)
                custom_chat_history.append((query_str, res.response))
                print(res.response)
                persist_context(custom_chat_history)
        except KeyboardInterrupt:
            print("Saving chat sessions...")
            persist_context(custom_chat_history)
            print("Chat sessions saved. Exiting...")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local LLM Domain Indexer")
    parser.add_argument('-l', '--load', action='store_true', help='Create LLM Index from local metafiles')
    parser.add_argument('-q', '--query', action='store_true', help='Query the LLM index')
    args = parser.parse_args()
    main(args)

import requests
import json
import time
import argparse
import boto3
from botocore.exceptions import NoCredentialsError
import os
import hashlib
from llama_index import SimpleDirectoryReader
from llama_index.chat_engine import CondenseQuestionChatEngine
from llama_index.chat_engine.types import ChatMode
from llama_index.vector_stores import OpensearchVectorStore, OpensearchVectorClient
from llama_index import VectorStoreIndex, StorageContext
from os import getenv
import sys
from llama_index.prompts import Prompt

# import open_clip
# from PIL import Image
# import torch


apiUrl = "https://api.videoindexer.ai"
accountId = "2c1aaab2-5675-4cc9-8de7-8815bd5723e0"
location = "trial"
apiKey = "5ae991260981469cb8a32ff915aeb2b2"


# model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
#     'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
# tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
#

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


def get_access_token(url):
    headers = {
        "Ocp-Apim-Subscription-Key": apiKey,
    }
    response = requests.get(url, headers=headers)
    access_token = response.text.replace("\"", "")
    return access_token


def get_account_access_token():
    url = f"{apiUrl}/auth/{location}/Accounts/{accountId}/AccessToken?allowEdit=true"
    return get_access_token(url)


def get_video_access_token(video_id):
    url = f"{apiUrl}/auth/{location}/Accounts/{accountId}/Videos/{video_id}/AccessToken?allowEdit=true"
    return get_access_token(url)


def upload_video_to_azure(video_url, account_access_token):
    response = requests.post(
        f"{apiUrl}/{location}/Accounts/{accountId}/Videos?accessToken={account_access_token}&name=some_name&description=some_description&privacy=private&partition=some_partition&videoUrl={video_url}")
    upload_result = response.text
    print(upload_result)
    video_id = json.loads(upload_result)["id"]
    return video_id


def reindex_video(video_id, account_access_token):
    response = requests.put(
        f"{apiUrl}/{location}/Accounts/{accountId}/Videos/{video_id}/Index?accessToken={account_access_token}&reIndex=true&sourceLanguage=ru-RU&indexingPreset=AdvancedVideo")
    reindex_result = response.text
    return reindex_result


def wait_for_video_index(video_id, video_access_token):
    while True:
        time.sleep(10)
        response = requests.get(
            f"{apiUrl}/{location}/Accounts/{accountId}/Videos/{video_id}/Index?accessToken={video_access_token}&language=en-US&indexingPreset=AdvancedVideo")
        video_get_index_result = response.text
        processing_state = json.loads(video_get_index_result)["state"]

        if processing_state != "Uploaded" and processing_state != "Processing":
            break


def get_video_index(video_id, video_access_token):
    response = requests.get(
        f"{apiUrl}/{location}/Accounts/{accountId}/Videos/{video_id}/Index?accessToken={video_access_token}&language=en-US")
    video_get_index_result = response.text
    return video_get_index_result


def update_dynamodb_azure_video_id(video_id, azure_video_id):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('argen-video-metadata')
    table.put_item(
        Item={
            'pk': video_id,
            'sk': 'azure',
            'azure_video_id': azure_video_id
        }
    )


def update_dynamodb_azure_result(video_id, video_index_result):
    # Convert metadata to JSON string
    metadata_json = video_index_result

    # Upload metadata to S3
    bucket_name = "artists-original-work"
    s3_object_name = f"{video_id}_metadata.json"
    s3 = boto3.client('s3')
    s3.put_object(Body=metadata_json, Bucket=bucket_name, Key=s3_object_name)
    s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_object_name}"

    # Save the S3 URL into the DynamoDB
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('argen-video-metadata')
    table.update_item(
        Key={
            'pk': video_id,
            'sk': 'azure'
        },
        UpdateExpression="SET video_index_result_url = :s3_url",
        ExpressionAttributeValues={
            ':s3_url': s3_url
        }
    )
    return s3_url


def get_video_metadata(provider, video_id):
    if provider == "azure":
        azure_video_id = get_azure_video_id(video_id)
        account_access_token = get_account_access_token()
        video_access_token = get_video_access_token(azure_video_id)
        video_index_result = get_video_index(azure_video_id, video_access_token)
        print(video_index_result)
        meta_url = update_dynamodb_azure_result(video_id, video_index_result)
        return meta_url, video_index_result
    elif provider == "aws":
        # Add AWS-specific code to get video metadata
        pass
    elif provider == "videochat":
        # Add VideoChat-specific code to get video metadata
        pass
    else:
        print("Invalid provider")


def compute_hash(local_file_path):
    with open(local_file_path, 'rb') as file:
        file_data = file.read()
        file_hash = hashlib.sha256(file_data).hexdigest()
    return file_hash


def get_azure_video_id(video_id):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('argen-video-metadata')
    response = table.get_item(
        Key={
            'pk': video_id,
            'sk': 'azure'
        }
    )
    print(response)
    return response['Item']['azure_video_id']


def create_job_json(video_id, meta_url):
    # content_id needs to be in format of : jobs/bb3pdt4xao0ib6kp/input_file/nasdaq-tsla-2022-10K-22595227.txt
    #         response = s3.get_object(Bucket=bucket, Key=job_info['file_url']['content_id'])
    job_json = {
        "job_id": video_id,
        "file_url": {
            "content_id": f"jobs/{video_id}/input_file/{video_id}.txt",
            "url": meta_url
        },
        "file_ext": "json",
        "created_time": int(time.time()),
        "username": "buryhuang",
        "app_name": "videochat"
    }
    return job_json


def store_job_json_to_s3(job_json, bucket_name, job_input):
    s3 = boto3.client('s3')
    job_id = job_json['job_id']

    # upload input files, forcing using txt, have to in the same bucket
    job_input_object_name = f"jobs/{job_id}/input_file/{job_id}.txt"
    s3.put_object(Body=job_input, Bucket=bucket_name, Key=job_input_object_name)

    # last step: upload job_meta to trigger the job
    job_meta_object_name = f"job_meta.json"
    s3.put_object(Body=json.dumps(job_json), Bucket=bucket_name, Key=f"jobs/{job_id}/{job_meta_object_name}")
    s3_url = f"https://{bucket_name}.s3.amazonaws.com/jobs/{job_id}/{job_meta_object_name}"
    return s3_url


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
    if args.images_folder_path:
        process_images_folder(args.images_folder_path)
        sys.exit(0)

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
        vector_store = OpensearchVectorStore(client)
        index = VectorStoreIndex.from_vector_store(vector_store)
        custom_chat_history = load_context()
        try:
            while True:
                query_str = input("Enter your question: ")
                query_engine = index.as_query_engine()

                chat_history_in_prompt = ""
                for (human_msg, agent_message) in custom_chat_history:
                    chat_history_in_prompt += f"Human: {human_msg}\nAgent: {agent_message}\n"

                custom_prompt = f"""\
                    Given a conversation (between Human and Assistant) and a follow up message from Human, \
                    continue the conversation that captures all relevant context \
                    from the chat history.
    
                    <Chat History> 
                    {chat_history_in_prompt}
    
                    <Follow Up Message>
                    {query_str}
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

    if args.chat:
        endpoint = getenv("OPENSEARCH_ENDPOINT", "http://localhost:9200")
        idx = getenv("OPENSEARCH_INDEX", "buryhuang-gpt-index-demo")
        client = OpensearchVectorClient(endpoint, idx, 1536, embedding_field="embedding", text_field="content")
        vector_store = OpensearchVectorStore(client)
        index = VectorStoreIndex.from_vector_store(vector_store)

        custom_prompt = Prompt("""\
        Given a conversation (between Human and Assistant) and a follow up message from Human, \
        continue the conversation that captures all relevant context \
        from the chat history.

        <Chat History> 
        {chat_history}

        <Follow Up Message>
        {question}

        """)

        # list of (human_message, ai_message) tuples
        custom_chat_history = load_context()

        query_engine = index.as_query_engine()
        chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine,
            condense_question_prompt=custom_prompt,
            chat_history=custom_chat_history,
            verbose=True
        )

        chat_history = []
        try:
            while True:
                query_str = input("\nYou: ")
                res = chat_engine.chat(query_str)
                custom_chat_history.append((query_str, res.response))
                print(res.response)
                persist_context(custom_chat_history)
        except KeyboardInterrupt:
            print("Saving chat sessions...")
            persist_context(chat_history)
            print("Chat sessions saved. Exiting...")

        sys.exit(0)

    if args.video_id:
        video_id = args.video_id
    else:
        if args.file:
            bucket_name = 'artists-original-work'
            file_path = args.file
            s3_url = upload_to_s3(file_path, bucket_name)
            video_url = s3_url
            video_id = compute_hash(args.file)
            print("Video uploaded to S3 with id: ", video_id)
        else:
            video_url = args.video_url

        account_access_token = get_account_access_token()
        azure_video_id = upload_video_to_azure(video_url, account_access_token)

        update_dynamodb_azure_video_id(video_id, azure_video_id)

        video_access_token = get_video_access_token(azure_video_id)
        wait_for_video_index(azure_video_id, video_access_token)

    meta_url, meta_blob = get_video_metadata(args.provider, video_id)

    if args.reindex:
        account_access_token = get_account_access_token()
        reindex_result = reindex_video(video_id, account_access_token)
        print("Reindexing started")
        print(reindex_result)

    if args.job_json:
        job_json = create_job_json(video_id, meta_url)
        job_json_s3_url = store_job_json_to_s3(job_json, "aigcc-llm-indexing", meta_blob)
        print("Job JSON stored at:", job_json_s3_url)


def process_image(image_path):
    # model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    # tokenizer = open_clip.get_tokenizer('ViT-B-32')

    text = tokenizer(["a diagram", "a dog", "a cat"])

    image = Image.open(image_path)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]


def process_images_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            print(process_image(file_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload video to Azure Video Indexer")
    parser.add_argument('-f', '--file', type=str, help='The path to the video file to be uploaded')
    parser.add_argument('-v', '--video_url', type=str, help='The video URL to analyze, if provided, skips S3 upload')
    parser.add_argument('-i', '--video_id', type=str,
                        help='The video ID to search, if provided, skips uploading and indexing')
    parser.add_argument('-r', '--reindex', action='store_true', help='Reindex the video using the given video_id')
    parser.add_argument('-p', '--provider', type=str, choices=["azure", "aws", "videochat"],
                        help='The provider to get video metadata from')
    parser.add_argument('-j', '--job_json', action='store_true', help='Create and store job JSON for the video')
    parser.add_argument('-l', '--load', action='store_true', help='Create LLM Index from local metafiles')
    parser.add_argument('-q', '--query', action='store_true', help='Query the LLM index')
    parser.add_argument('-c', '--chat', action='store_true', help='Chat with the LLM index')
    parser.add_argument('-d', '--images_folder_path', type=str, help='The path to the folder containing the images')
    args = parser.parse_args()
    main(args)

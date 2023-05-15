from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import openai
import json
import numpy as np
from numpy.linalg import norm
import re
from time import time,sleep
from uuid import uuid4
import datetime
import pinecone

load_dotenv()

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)


def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")


model_name = os.getenv("MODEL_NAME")

def emedding(content, model_name=model_name):
    content = content.encode(encoding='ASCII',errors='ignore').decode()  # fix any UNICODE errors
    response = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=openai.api_key
    )
    vector = response['data'][0]['embedding']  
    return vector

# def gpt3_embedding(content, engine='text-embedding-ada-002'):
#     content = content.encode(encoding='ASCII',errors='ignore').decode()  # fix any UNICODE errors
#     response = openai.Embedding.create(input=content,engine=engine)
#     vector = response['data'][0]['embedding']  # this is a normal list
#     return vector




GPT_VERSION = os.getenv("GPT_VERSION")

def gpt_completion(prompt, model_name=GPT_VERSION, temp=0.0, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0, stop=['USER:', 'MED:']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = ChatOpenAI(
                    openai_api_key=openai.api_key,
                    model_name=GPT_VERSION,
                    temperature=temp,
                    max_tokens=tokens,
                    top_p=top_p,
                    frequency_penalty=freq_pen,
                    presence_penalty=pres_pen,
                    stop=stop
            )
            text = response['choices'][0]['text'].strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            if not os.path.exists('gpt3_logs'):
                os.makedirs('gpt3_logs')
            save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)
        


def load_conversation(results):
    result = list()
    for m in results['matches']:
        info = load_json('nexus/%s.json' % m['id'])
        result.append(info)
    ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
    messages = [i['message'] for i in ordered]
    return '\n'.join(messages).strip()


if __name__ == '__main__':
    convo_length = 30
    openai.api_key = os.getenv("OPENAI_API_KEY")
    YOUR_API_KEY = os.getenv("PINECONE_API_KEY")
    YOUR_ENV = os.getenv("ENVIRONMENT")
    INDEX_NAME = os.getenv("INDEX_NAME")

    pinecone.init(
    api_key=YOUR_API_KEY,
    environment=YOUR_ENV)

    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(
            name=INDEX_NAME,
            metric='dotproduct',
            dimension=1536)
        
    index = pinecone.Index(INDEX_NAME)
    
    while True:
        #### get user input, save it, vectorize it, save to pinecone
        payload = list()
        timestamp = time()
        timestring = timestamp_to_datetime(timestamp)
        message = input('\n\nUSER: ')
        vector = gpt_completion(message)
        unique_id = str(uuid4())
        metadata = {'speaker': 'USER', 'time': timestamp, 'message': message, 'timestring': timestring, 'uuid': unique_id}
        save_json('nexus/%s.json' % unique_id, metadata)
        payload.append((unique_id, vector))
        #### search for relevant messages, and generate a response
        results = index.query(vector=vector, top_k=convo_length)
        
        # FIGURE out how compares to Conversation memory and retrieval QA, figure out if retrievalQA weight retrieving too heavily
        conversation = load_conversation(results)  # results should be a DICT with 'matches' which is a LIST of DICTS, with 'id'
        
        prompt = open_file('prompt_response.txt').replace('<<CONVERSATION>>', conversation).replace('<<MESSAGE>>', message)
        #### generate response, vectorize, save, etc
        output = gpt_completion(prompt)
        timestamp = time()
        timestring = timestamp_to_datetime(timestamp)
        #message = '%s: %s - %s' % ('RAVEN', timestring, output)
        message = output
        vector = gpt_completion(message)
        unique_id = str(uuid4())
        metadata = {'speaker': 'MED', 'time': timestamp, 'message': message, 'timestring': timestring, 'uuid': unique_id}
        save_json('nexus/%s.json' % unique_id, metadata)
        payload.append((unique_id, vector))
        index.upsert(payload)
        print('\n\MED: %s' % output) 
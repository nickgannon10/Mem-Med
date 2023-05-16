from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from prompt import jsonify_prompts,
from dotenv import load_dotenv
from numpy.linalg import norm
from time import time,sleep
from uuid import uuid4
import numpy as np
import datetime
import pinecone
import openai
import json
import os
import re

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

def print_json_file(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
        print(json.dumps(data, indent=4))


def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")


model_name = os.getenv("MODEL_NAME")

# def emedding(content, model_name=model_name):
#     content = content.encode(encoding='ASCII',errors='ignore').decode()  # fix any UNICODE errors
#     response = OpenAIEmbeddings(
#         model=model_name,
#         openai_api_key=openai.api_key
#     )
#     vector = response['data'][0]['embedding']  
#     return vector

GPT_VERSION = os.getenv("GPT_VERSION")

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

def gpt_completion(prompt, model_name=GPT_VERSION, conversational_memory=conversational_memory, temp=0.0, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0, stop=['USER:', 'MED:']):
    while True:
        # try:
        llm = ChatOpenAI(
                openai_api_key=openai.api_key,
                model_name=GPT_VERSION,
                temperature=temp
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        tools = [
            Tool(
                name='Knowledge Base',
                func=qa.run,
                description=(
                    'Do not use this tool, it does not have any relevant information about the topic.'
                )
            )
        ]

        agent = initialize_agent(
            agent='chat-conversational-react-description',
            tools=tools,
            llm=llm,
            verbose=True,
            max_iterations=3,
            early_stopping_method='generate',
            memory=conversational_memory
        )

        response = agent(prompt)

        conversational_memory.save_context({"input": prompt}, {"output": response['output']})

        response['chat_history'] = conversational_memory.load_memory_variables({})['chat_history']

        return response


# def load_conversation(results):
#     result = list()
#     for m in results['matches']:
#         info = load_json('nexus/%s.json' % m['id'])
#         result.append(info)
#     ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
#     messages = [i['message'] for i in ordered]
#     return '\n'.join(messages).strip()

batch_messages = []
if __name__ == '__main__':
    convo_length = 30
    openai.api_key = os.getenv("OPENAI_API_KEY")
    YOUR_API_KEY = os.getenv("PINECONE_API_KEY")
    YOUR_ENV = os.getenv("ENVIRONMENT")
    INDEX_NAME = os.getenv("INDEX_NAME")
    TEXT_FIELD = os.getenv("TEXT_FIELD")

    pinecone.init(
        api_key=YOUR_API_KEY,
        environment=YOUR_ENV
    )

    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(
            name=INDEX_NAME,
            metric='dotproduct',
            dimension=1536)
        
    index = pinecone.Index(INDEX_NAME)

    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=openai.api_key
    )

    vectorstore = Pinecone(
        index, embed.embed_query, TEXT_FIELD
    )

    while True:
#         #### get user input, save it, vectorize it, save to pinecone
#         payload = list()
        timestamp = time()
        timestring = timestamp_to_datetime(timestamp)

        message = input('\n\nUSER: ')
        formatted_prompt = jsonify_prompts(message)
        
        
        vectorstore.similarity_search(
            message,  # our search query
            k=3  # return 3 most relevant docs
        )
        vector = gpt_completion(formatted_prompt)



#         unique_id = str(uuid4())
#         metadata = {'speaker': 'USER', 'time': timestamp, 'message': message, 'timestring': timestring, 'uuid': unique_id}
#         save_json('nexus/%s.json' % unique_id, metadata)
#         payload.append((unique_id, vector))
#         #### search for relevant messages, and generate a response
#         results = index.query(vector=vector, top_k=convo_length)
        
        
#         # FIGURE out how compares to Conversation memory and retrieval QA, figure out if retrievalQA weight retrieving too heavily
#         conversation = load_conversation(results)  # results should be a DICT with 'matches' which is a LIST of DICTS, with 'id'
#         # also integrate best practices from orderbot

        # prompt = open_file('prompt_response.txt').replace('<<CONVERSATION>>', conversation).replace('<<MESSAGE>>', message)
#         #### generate response, vectorize, save, etc
        # output = gpt_completion(prompt)
#         timestamp = time()
#         timestring = timestamp_to_datetime(timestamp)
#         #message = '%s: %s - %s' % ('RAVEN', timestring, output)
#         message = output
#         vector = gpt_completion(message)
#         unique_id = str(uuid4())
#         metadata = {'speaker': 'MED', 'time': timestamp, 'message': message, 'timestring': timestring, 'uuid': unique_id}
#         save_json('nexus/%s.json' % unique_id, metadata)
#         payload.append((unique_id, vector))
#         index.upsert(payload)
        print('\n\nMED: %s' % vector) 
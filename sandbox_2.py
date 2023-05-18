from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from prompt import jsonify_prompts, timestamp_to_datetime, append_to_json
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.agents import Tool
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
GPT_VERSION = os.getenv("GPT_VERSION")

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    return_messages=True
)

def gpt_completion(prompt, model_name=GPT_VERSION, conversational_memory=conversational_memory, temp=0.0, stop=['USER:', 'MED:']):
    while True: 
        llm = ChatOpenAI(
            openai_api_key=openai.api_key,
            model_name=GPT_VERSION,
            temperature=temp
        )

        qa = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0), 
            vectorstore.as_retriever(), 
            memory=conversational_memory
        )
        with get_openai_callback() as cb:
            response = qa({"question": prompt})
            total_tokens = cb.total_tokens

        return response, total_tokens, conversational_memory

def append_output_json(prompt_template, vector, total_tokens):
    chat_history = []
    for message in vector['chat_history']:
        chat_history.append(message.__dict__)
    prompt_template.__dict__["output_parser"] = {
        "question": vector['question'], 
        "chat_history": chat_history, 
        "answer": vector['answer'],
        "total_tokens": total_tokens
    }
    return prompt_template

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
            dimension=1536
        )
        
    index = pinecone.Index(INDEX_NAME)

    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=openai.api_key
    )

    vectorstore = Pinecone(
        index, embed.embed_query, TEXT_FIELD
    )

    while True:

        message = input('\n\nUSER: ')
        formatted_prompt = jsonify_prompts(user_input=message)
        prep_prompt = formatted_prompt.format(user_input=message)

        vectorstore.similarity_search(
            message,  # our search query
            k=3  # return 3 most relevant docs
        )

        vector, total_tokens, conversational_memory = gpt_completion(prep_prompt)

        append_output_json(formatted_prompt, vector, total_tokens)

        append_to_json("sandbox_2.json", formatted_prompt.__dict__)

        print('\n\nRAVEN_QUESTION: %s' % vector['question'])
        print('\n\nRAVEN_CHAT_HISTORY: %s' % vector['chat_history'])
        print('\n\nRAVEN_ANSWER: %s' % vector['answer'])
        print('\n\nRAVEN_conversation_buffer_memory: %s' % conversational_memory)

        # 1. after gpt_completetion is generated, we want to append the output into the json
            # file in the "output parser" 
            # something like this:
            # "output parser": {
            #    "output": "someoutput" 
            #    "chat_history": []
            #    "full_conversational_buffer_memory": []
            #    "token_count_of_conversational_buffer_memory": 1200
            # }

        # 2. next we want to do two things: 
            # 1. re-format this information with the data and metadata that pinecone will accept
            # 2. embed that re-formatted chunk into a pinecone index

        # 3. We want to load the conversation memory back into apon re-starting such 
        # such that the conversation neatly continues where it left off



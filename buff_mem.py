from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationChain
from langchain import OpenAI
from dotenv import load_dotenv
import openai
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# first initialize the large language model
llm = OpenAI(
	temperature=0,
	openai_api_key=openai.api_key,
	model_name="text-davinci-003"
)

# now initialize the conversation chain
conversation = ConversationChain(llm=llm)

# print(conversation.prompt.template)

conversation_buf = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result

count_tokens(
    conversation_buf, 
    "My interest here is to explore the potential of integrating Large Language Models with external knowledge"
)

print(conversation_buf.memory.buffer)

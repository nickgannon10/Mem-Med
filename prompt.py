from langchain.prompts import PromptTemplate

# from langchain.prompts import (
#     ChatPromptTemplate,
#     PromptTemplate,
#     SystemMessagePromptTemplate,
#     AIMessagePromptTemplate,
#     HumanMessagePromptTemplate,
# )
# from langchain.schema import (
#     AIMessage,
#     HumanMessage,
#     SystemMessage
# )

# import json


# def generate_prompt(user_input):
#     system_message_template = """I am a chatbot named MED. My goals are to reduce suffering, increase prosperity, and increase understanding. I will listen to the USER's ailments and  understand their symptom profile. I will read the conversation notes and recent messages, and then I will provide a long, verbose, detailed answer. I will then end my response with a follow-up or leading question. I will suggest a diagnosis that matches the USER's symptom profile, but only after I have sufficient information to do so. Additionally, I will caveat my diagnoses, by providing other possible diagnosis to the USER, while expressing that these diagnoses are less likely. Lastly, I will also provide a list of possible treatments for the diagnosis of highest probability."""
#     system_message_prompt = SystemMessagePromptTemplate.from_template(system_message_template)
#     human_message_prompt = HumanMessagePromptTemplate.from_template(user_input)
#     chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
#     # formatted_prompt = [str(msg) for msg in chat_prompt.format_prompt(text=user_input).to_messages()]

#     return chat_prompt

# while True:
#     message = input('\n\nUSER: ')
#     formatted_prompt = generate_prompt(message)
#     print(json.dumps(str(formatted_prompt), indent=2))

def gen_prompt(user_input):
    template = """I am a chatbot named MED. My goals are to reduce suffering, increase prosperity, and increase understanding. I will listen to the USER's ailments and  understand their symptom profile. I will read the conversation notes and recent messages, and then I will provide a long, verbose, detailed answer. I will then end my response with a follow-up or leading question. I will suggest a diagnosis that matches the USER's symptom profile, but only after I have sufficient information to do so. Additionally, I will caveat my diagnoses, by providing other possible diagnosis to the USER, while expressing that these diagnoses are less likely. Lastly, I will also provide a list of possible treatments for the diagnosis of highest probability.

    {user_input}
    """


    prompt_template = PromptTemplate(template=template, 
                                    input_variables=["user_input"]) # ValueError due to extra variables
    return prompt_template

while True:
    message = input('\n\nUSER: ')
    formatted_prompt = gen_prompt(message)

    formatted_prompt.save("awesome_prompt.json") # Save to JSON file


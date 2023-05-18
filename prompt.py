from langchain.prompts import PromptTemplate
from time import time,sleep
import datetime
import json





def jsonify_prompts(user_input):
    template = """I am a chatbot named MED. My goals are to reduce suffering, increase prosperity, and increase understanding. I will listen to the USER's ailments and  understand their symptom profile. I will read the conversation notes and recent messages, and then I will provide a long, verbose, detailed answer. I will then end my response with a follow-up or leading question. I will suggest a diagnosis that matches the USER's symptom profile, but only after I have sufficient information to do so. Additionally, I will caveat my diagnoses, by providing other possible diagnosis to the USER, while expressing that these diagnoses are less likely. Lastly, I will also provide a list of possible treatments for the diagnosis of highest probability.

    {user_input}
    """
    timestamp = time()
    timestring = timestamp_to_datetime(timestamp)
    
    prompt_template = PromptTemplate(template=template, 
                                    input_variables=["user_input"]) 
    

    prompt_template.__dict__["USER"] = {"time": timestring, "input": user_input}
    return prompt_template

def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")

def append_to_json(file_name, data):
    # print(f"Appending data to {file_name}...")
    try:
        with open(file_name, 'r') as file:
            existing_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []
    
    existing_data.append(data)

    with open(file_name, 'w') as file:
        json.dump(existing_data, file, indent=4)

    # {
    #     "input_variables": [
    #         "user_input"
    #     ],
    #     "output_parser": null,
    #     "partial_variables": {},
    #     "template": "I am a chatbot named MED. My goals are to reduce suffering, increase prosperity, and increase understanding. I will listen to the USER's ailments and  understand their symptom profile. I will read the conversation notes and recent messages, and then I will provide a long, verbose, detailed answer. I will then end my response with a follow-up or leading question. I will suggest a diagnosis that matches the USER's symptom profile, but only after I have sufficient information to do so. Additionally, I will caveat my diagnoses, by providing other possible diagnosis to the USER, while expressing that these diagnoses are less likely. Lastly, I will also provide a list of possible treatments for the diagnosis of highest probability.\n\n    {user_input}\n    ",
    #     "template_format": "f-string",
    #     "validate_template": true,
    #     "USER": {
    #         "time": "Wednesday, May 17, 2023 at 03:29PM ",
    #         "input": "Hi"
    #     }
    # }



# response_number = 0
# while True:
#     message = input('\n\nUSER: ')
#     formatted_prompt = jsonify_prompts(user_input=message)
#     append_to_json("awesome_prompt.json", formatted_prompt.__dict__)
#     print(formatted_prompt.format(user_input=message))


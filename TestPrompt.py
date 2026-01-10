import os
import openai
# from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
# from langchain.chat_models import ChatOpenAI


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2026, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"




# client = OpenAI()  # Assumes OPENAI_API_KEY is already in env

""" def get_completion(prompt, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content


output = get_completion("What is 1+1?")
print(output) """



llm = ChatOpenAI(model="gpt-4.1")
# response = llm.invoke("What is 1+1?")
# print(response.content)

template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""

""" prompt = PromptTemplate.from_template(template_string)
prompt_value = prompt.format(style="Shakespearean", text="To be or not to be.")
response = llm.invoke(prompt_value)
print(response.content)  """



prompt = PromptTemplate.from_template(
    """
You are an expert English teacher. Your job is to evaluate the student's answer and give helpful feedback.

**Task:**
- Read the question and the student's answer.
- Decide if the answer is correct, partially correct, or incorrect.
- Give clear feedback focusing on grammar, clarity, vocabulary, and content.
- Provide a better version only if necessary.

**Output Format:**
Evaluation:
- Correctness: (Correct / Partially correct / Incorrect)
- Score: X/10
- Strengths: ...
- Areas to improve: ...
- Suggested improved answer: ...

**Here is the content to evaluate:**
Question: "{question}"
Student Answer: "{answer}"
"""
)

prompt_value = prompt.format(
    question="Why is the sky blue?",
    answer="Because the ocean is reflected in the sky."
)
response = llm.invoke(prompt_value)
print(response.content)
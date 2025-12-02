from dotenv import load_dotenv
from langchain_openai import OpenAI

load_dotenv()

llm = OpenAI(model='gpt-3.5-turbo-instruct')

prompt = f'What is the capital of India ?'
response = llm.invoke(prompt)
print(response)
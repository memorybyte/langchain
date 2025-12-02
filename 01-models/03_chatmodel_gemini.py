from pprint import pprint
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
prompt = f'What is the capital of India ?'
response = model.invoke(prompt)

pprint(f'Response Type: {type(response)}')
pprint(f'Response Content: {response.content}')
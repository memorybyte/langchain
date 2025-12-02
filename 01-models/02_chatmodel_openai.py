from pprint import pprint
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model='gpt-4.1-nano',
    temperature=2, # Randomness
    # max_completion_tokens=10 # Limits the maximum number of tokens in the model's response
)
prompt = f'What is the capital of India ?'
response = model.invoke(prompt)

pprint(f'Response Type: {type(response)}')
pprint(f'Response Content: {response.content}')
from pprint import pprint
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

model = ChatAnthropic(model='claude-sonnet-4-5')
prompt = f'What is the capital of India ?'
response = model.invoke(prompt)

pprint(f'Response Type: {type(response)}')
pprint(f'Response Content: {response.content}')
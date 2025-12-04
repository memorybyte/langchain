from pprint import pprint
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# model = ChatOpenAI(model='gpt-4.1-nano')
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# With chat history
messages = [
    SystemMessage(content='You are a helpful AI Assistant.'),
    HumanMessage(content='Tell me about LangChain in 50 words.')
]

response = model.invoke(messages)

messages.append(AIMessage(content=response.content))

pprint(messages)
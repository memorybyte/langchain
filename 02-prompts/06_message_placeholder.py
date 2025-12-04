from pprint import pprint
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# model = ChatOpenAI(model='gpt-4.1-nano')
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# Chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent.'),
    MessagesPlaceholder(variable_name='chat_history'), # Previous chat
    ('human', '{query}')
])

chat_history = []
# Load Chat History
with open('chat_history.txt', 'r') as f:
    chat_history.extend(f.readlines())

# Create prompt
prompt = chat_template.invoke({
    'chat_history': chat_history,
    'query': 'Where is my refund ?'
    })

pprint(prompt)

from pprint import pprint
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# model = ChatOpenAI(model='gpt-4.1-nano')
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# This method does not works
# chat_template = ChatPromptTemplate([
#     SystemMessage(content='You are a helpful {domain} expert.'),
#     HumanMessage(content='Explain me simple terms what is {topic}')
# ])


chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert.'),
    ('human', 'Explain me simple terms what is {topic}')
])

prompt = chat_template.invoke({
    'domain': 'cricket',
    'topic': 'dusra'
},)

print(prompt)
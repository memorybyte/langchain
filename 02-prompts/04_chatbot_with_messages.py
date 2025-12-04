from pprint import pprint
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# model = ChatOpenAI(model='gpt-4.1-nano')
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')


chat_history = [
    SystemMessage(content='You are a helpful AI Assistant.')
]
while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print(f'AI: {result.content}\n\n')

print(f'\n\n')
pprint(chat_history)
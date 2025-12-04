from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# model = ChatOpenAI(model='gpt-4.1-nano')
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')


# 1. Basic
# while True:
#     user_input = input('You: ')
#     if user_input == 'exit':
#         break
#     result = model.invoke(user_input)
#     print(f'AI: {result.content}')


# 2. With chat history
chat_history = []
while True:
    user_input = input('You: ')
    chat_history.append(user_input)
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(result.content)
    print(f'AI: {result.content}')
# import os
# from dotenv import load_dotenv
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# load_dotenv()

# hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
# if hf_token and not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
#     os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
# )

# model = ChatHuggingFace(llm=llm)

# prompt = f'Who is the captain of Men"s Indian Cricket Team ?'
# response = model.invoke(prompt)
# print(response.content)


import os
from langchain_huggingface import HuggingFaceEndpoint

# Ensure the environment variable is set or pass it directly
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_API_TOKEN" 

repo_id = "mistralai/Mistral-Nemo-Instruct-2407" # Example model
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens= 250, 
    temperature= 0.7
)

# Use the LLM object for text generation
prompt = "Explain how LangChain uses the HuggingFaceEndpoint"
response = llm.invoke(prompt) 
print(response)

# The model can also be used in a LangChain Expression Language (LCEL) chain
# chain = prompt_template | llm
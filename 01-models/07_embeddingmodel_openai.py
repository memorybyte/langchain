from pprint import pprint 
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embedding = OpenAIEmbeddings(
    model='text-embedding-3-small',
    dimensions=32
)

# Embed single query
text = 'Delhi is the capital of India.'
# embedded_text = embedding.embed_query(text)
# print(len(embedded_text), str(embedded_text))

# Embed multiple query
texts = [
    'Delhi is the capital of India',
    'Kolkata is the capital of West Bengal.'
]

embedded_documents = embedding.embed_documents(texts)
pprint(embedded_documents)
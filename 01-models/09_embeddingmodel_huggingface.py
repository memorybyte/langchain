from pprint import pprint
from langchain_huggingface import HuggingFaceEmbeddings

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
embedding = HuggingFaceEmbeddings(model=model_name)

# Embed single query
text = 'Delhi is the capital of India.'
embedded_text = embedding.embed_query(text)
print(len(embedded_text))
pprint(embedded_text)
print('\n\n')

# Embed multiple query
texts = [
    'Delhi is the capital of India',
    'Kolkata is the capital of West Bengal.'
]

embedded_documents = embedding.embed_documents(texts)
pprint(embedded_documents)
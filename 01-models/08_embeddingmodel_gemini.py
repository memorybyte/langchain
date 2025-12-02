from dotenv import load_dotenv
from pprint import pprint
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

query_text = "What is the capital of France?"
query_vector = embeddings.embed_query(query_text)

print(f"Query Vector Length: {len(query_vector)}")


document_texts = [
    "Paris is known as the City of Love and the capital of France.",
    "A computer is an electronic device that processes data.",
    "The sun is a star at the center of the solar system."
]

document_vectors = embeddings.embed_documents(document_texts)

print(f"Number of document vectors: {len(document_vectors)}")
print(f"Vector length of first document: {len(document_vectors[0])}")
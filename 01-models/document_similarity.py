from pprint import pprint
from dotenv import load_dotenv
import numpy
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)
embeddings = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')


summaries = [
    "MS Dhoni: Former Indian cricket team captain, renowned for his calm demeanor and exceptional leadership. Led India to victories in the 2007 T20 World Cup, 2011 ODI World Cup, and 2013 Champions Trophy. Known for his finishing skills and wicketkeeping.",
    "Virat Kohli: One of the world’s leading batsmen, known for his aggressive style and consistency across formats. Former Indian captain, he holds numerous batting records and is celebrated for his fitness and passion for the game.",
    "Rohit Sharma: Current Indian team captain (as of 2025), famous for his elegant batting and record three double centuries in ODIs. A prolific opener, he has led Mumbai Indians to multiple IPL titles.",
    "Sachin Tendulkar: Widely regarded as one of the greatest batsmen in cricket history. Holds the record for most runs in international cricket and 100 international centuries. Revered as the 'God of Cricket' in India.",
    "Suresh Raina: Known for his aggressive batting and excellent fielding, Raina was a key middle-order player for India. He was the first Indian to score a century in all three formats and played a crucial role in India’s 2011 World Cup win."
]

query = 'Tell me about suresh raina'

document_embedding = embeddings.embed_documents(summaries)
query_embedding = embeddings.embed_query(query)

scores = cosine_similarity([query_embedding], document_embedding)[0]
scores = list(enumerate(scores))
sorted_score = sorted(scores, key=lambda x: x[1])
print(sorted_score)

relevant_document_index, _ = sorted_score[-1]
print(f'\nQuery: {query}')
print(f'Answer: {summaries[relevant_document_index]}')
from pprint import pprint
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# model = ChatOpenAI(model='gpt-4.1-nano')
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')


# Schema
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}


model_with_structured_output = model.with_structured_output(json_schema)

# prompt = (
#     'The hardware is great, but the software feels bloated. '
#     'There are too many pre-installed apps that I can not remove. '
#     'Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.'
#     )

prompt = (
    'I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it is an absolute powerhouse!' 
    'The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I am gaming, multitasking, or editing photos.'
    'The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.'
    'The S-Pen integration is a great touch for note-taking and quick sketches, though I donot use it often.'
    'What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light.'
    'Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.'
    'However, the weight and size make it a bit uncomfortable for one-handed use.' 
    'Also, Samsungs One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides?' 
    'The $1,300 price tag is also a hard pill to swallow.'
    'Pros:'
    'Insanely powerful processor (great for gaming and productivity)'
    'Stunning 200MP camera with incredible zoom capabilities'
    'Long battery life with fast charging'
    'S-Pen support is unique and useful'

    'Review by Memory Byte'
)

result = model_with_structured_output.invoke(prompt)

pprint(result)
print('\n\n')
print(type(result)) 
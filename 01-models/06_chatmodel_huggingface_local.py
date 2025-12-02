from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

model_id = 'HuggingFaceH4/zephyr-7b-beta'

llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task='chat-completion',
    pipeline_kwargs={
        'temperature': 0.5,
        'max_new_tokens': 100
    }
)

model = ChatHuggingFace(llm=llm)

prompt = f'Who is the captain of Men"s Indian Cricket Team ?'
response = model.invoke(prompt)
print(response.content)

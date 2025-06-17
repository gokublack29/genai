from transformers import pipeline
qa_pipeline = pipeline("question-answering")
def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']
context = "Last night, there was a massive power outage in New York, causing the city's lights to go out."
question = "What happened in New York last night?"
print(answer_question(question, context))
generator = pipeline("text-generation", model="gpt2")
def generate_text(prompt, max_length=50):
    result = generator(prompt, max_length=max_length, num_return_sequences=1)
    return result[0]['generated_text']
prompt = "Once upon a time, in a distant land,"
generated_text = generate_text(prompt)
print(generated_text)

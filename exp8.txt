# Different Prompts for Named Entity Recognition

# 1. Direct Prompting
"Identify named entities (persons, organizations, locations) in the following sentence:
'Elon Musk, the CEO of Tesla, visited Berlin to discuss business opportunities with German officials.'"

# 2. Few-shot Prompting
"Extract named entities from the sentences below:
'Apple was founded by Steve Jobs in California.' -> Apple (ORG), Steve Jobs (PERSON), California (LOC)
'Barack Obama served as the 44th President of the United States.' -> Barack Obama (PERSON), United States (LOC)
'Microsoft announced a new AI model at their Redmond headquarters.' -> Microsoft (ORG), Redmond (LOC)"

# 3. Chain-of-Thought Prompting
"Let's analyze the following sentence step by step and extract named entities:
'Sundar Pichai, CEO of Google, spoke at the AI conference in London on July 15, 2024.'
Identify the person(s).
Identify the organization(s).
Identify the location(s).
Identify any date(s).
Return the extracted entities in a structured format."

# 4. Role-based Prompting
"You are an expert in Named Entity Recognition. Your task is to extract and classify named entities from the given text.
Text: 'NASA announced a new space mission to Mars, scheduled for 2025.'
Return the named entities along with their categories."

# 5. Structured Prompting
"Extract and categorize named entities from the following text:
'Jeff Bezos, the founder of Amazon, attended a space launch event in Florida on June 5, 2023.'
Return the result in JSON format:
{
    'Person': [],
    'Organization': [],
    'Location': [],
    'Date': []
}"

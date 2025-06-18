# Different Prompts for Sentiment Analysis

# 1. Direct Prompting
"Analyze the sentiment of this text: 'I love the new phone! It has an amazing camera and great battery life.'"

# 2. Few-shot Prompting
"Classify the sentiment of the following sentences:
'The movie was absolutely terrible and a waste of time.' -> Negative
'I had an okay experience at the restaurant, nothing special.' -> Neutral
'This product is fantastic! I highly recommend it.' -> Positive"

# 3. Chain-of-Thought Prompting
"Let's analyze the sentiment of this statement step by step:
'I was really excited about this product, but after using it, I feel disappointed.'
First, identify the emotions expressed.
Then, determine if the overall sentiment is positive, negative, or neutral.
Finally, conclude with a sentiment label."

# 4. Role-based Prompting
"You are an expert sentiment analysis AI. Your task is to classify the sentiment of the given text. Respond with Positive, Negative, or Neutral.
Text: 'I regret buying this item. It broke within two days.'"

# 5. Structured Prompting
"Analyze the sentiment of the given review and return a structured JSON output.
Text: 'This hotel had excellent service, but the rooms were too small and noisy.'
Output format:
{
    'Sentiment': 'Positive/Negative/Neutral',
    'Reason': 'Explanation for the classification'
}"

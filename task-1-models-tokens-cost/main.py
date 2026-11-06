

import tiktoken
import json

# Load the tokenizer
encoding = tiktoken.get_encoding("cl100k_base")

# Different types of text
texts = {
    "Sentence": "Artificial intelligence is changing the way people work.",

    "Paragraph": "Large language models are trained on huge amounts of text data. They understand patterns in language and generate human-like responses. These models are useful for education, business, research and automation.",

    "Page": "Foundation models are large machine learning models trained on broad datasets. They can perform many tasks such as summarization, translation, question answering and content generation. Tokens are the basic units processed by language models. A token may represent a complete word, part of a word, punctuation mark or character. Different models use different tokenizers, so the same sentence may produce different token counts. Context windows determine how much input and output a model can handle in one interaction. A larger context window is useful when working with long documents, reports and conversations. Developers must also consider model quality, response speed, privacy and API costs.",

    "JSON": json.dumps({
        "name": "AI Assistant",
        "type": "language model",
        "features": ["chat", "summarization", "translation"],
        "active": True,
        "users": 1000
    }),

    "Hindi": "कृत्रिम बुद्धिमत्ता दुनिया को बदल रही है।"
}

# Count words and tokens
print("\n===== TOKEN COUNT RESULTS =====")

print("Text\t\tWords\tTokens\tWords/Token")

results = {}

for name in texts:

    text = texts[name]

    # Count words
    words = len(text.split())

    # Count tokens
    tokens = len(encoding.encode(text))

    # Calculate words per token
    ratio = words / tokens

    results[name] = tokens

    print(name, "\t", words, "\t", tokens, "\t", round(ratio, 3))


# Compare context windows
print("\n===== CONTEXT WINDOW COMPARISON =====")

for name in results:

    tokens = results[name]

    if tokens <= 8192:
        result8 = "YES"
    else:
        result8 = "NO"

    if tokens <= 131072:
        result128 = "YES"
    else:
        result128 = "NO"

    print(name, "  8K:", result8, "  128K:", result128)


# Calculate monthly API cost
print("\n===== MONTHLY API COST =====")

chats = 1000
input_tokens = 1500
output_tokens = 400
days = 30

# Calculate total monthly tokens
total_input = chats * input_tokens * days
total_output = chats * output_tokens * days

# Example API prices per 1 million tokens
input_price = 0.30
output_price = 2.50

# Calculate input cost
input_cost = (total_input / 1000000) * input_price

# Calculate output cost
output_cost = (total_output / 1000000) * output_price

# Calculate total cost
total_cost = input_cost + output_cost

# Display results
print("Chats per day:", chats)
print("Monthly input tokens:", total_input)
print("Monthly output tokens:", total_output)

print("Input price: $", input_price)
print("Output price: $", output_price)

print("Total monthly cost: $", round(total_cost, 2))

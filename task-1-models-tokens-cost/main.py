
import tiktoken
import json

# Load GPT-compatible tokenizer
encoding = tiktoken.get_encoding("cl100k_base")


# Five different types of text
texts = {
    "Sentence": "Artificial intelligence is changing the way people work.",

    "Paragraph": (
        "Large language models are trained on huge amounts of text data. "
        "They understand patterns in language and generate human-like responses. "
        "These models are useful for education, business, research and automation."
    ),

    "Page": (
        "Foundation models are large machine learning models trained on broad "
        "datasets. They can perform many tasks such as summarization, translation, "
        "question answering and content generation. Tokens are the basic units "
        "processed by language models. A token may represent a complete word, "
        "part of a word, punctuation mark or character. Different models use "
        "different tokenizers, so the same sentence may produce different token "
        "counts. Context windows determine how much input and output a model can "
        "handle in one interaction. A larger context window is useful when working "
        "with long documents, reports and conversations. Developers must also "
        "consider model quality, response speed, privacy and API costs."
    ),

    "JSON": json.dumps({
        "name": "AI Assistant",
        "type": "language model",
        "features": ["chat", "summarization", "translation"],
        "active": True,
        "users": 1000
    }, ensure_ascii=False),

    "Hindi": "कृत्रिम बुद्धिमत्ता दुनिया को बदल रही है।"
}


# Count words
def count_words(text):
    return len(text.split())


# Count tokens
def count_tokens(text):
    return len(encoding.encode(text))


# Calculate API cost
def estimate_cost(input_tokens, output_tokens, price_in, price_out):
    """
    Prices are in USD per 1 million tokens.
    """
    input_cost = (input_tokens / 1_000_000) * price_in
    output_cost = (output_tokens / 1_000_000) * price_out

    return input_cost + output_cost


# Display token counting results
print("\n========== TOKEN COUNT RESULTS ==========")

print(f"{'Text':<15} {'Words':<10} {'Tokens':<10} {'Words/Token':<12}")

results = {}

for name, text in texts.items():

    words = count_words(text)
    tokens = count_tokens(text)

    ratio = words / tokens if tokens > 0 else 0

    results[name] = tokens

    print(f"{name:<15} {words:<10} {tokens:<10} {ratio:.3f}")


# Compare 8K and 128K context windows
print("\n========== CONTEXT WINDOW COMPARISON ==========")

for name, tokens in results.items():

    fits_8k = tokens <= 8192
    fits_128k = tokens <= 131072

    print(
        f"{name:<15} "
        f"8K: {'YES' if fits_8k else 'NO':<5} "
        f"128K: {'YES' if fits_128k else 'NO'}"
    )


# Monthly API cost estimation
print("\n========== MONTHLY COST ESTIMATE ==========")

chats_per_day = 1000
input_per_chat = 1500
output_per_chat = 400
days_per_month = 30

monthly_input = chats_per_day * input_per_chat * days_per_month

monthly_output = chats_per_day * output_per_chat * days_per_month

# Example Gemini 2.5 Flash standard text pricing
price_in = 0.30
price_out = 2.50

monthly_cost = estimate_cost(
    monthly_input,
    monthly_output,
    price_in,
    price_out
)

print("Chats per day:", chats_per_day)
print("Monthly input tokens:", monthly_input)
print("Monthly output tokens:", monthly_output)

print("Input price per 1M tokens: $", price_in)
print("Output price per 1M tokens: $", price_out)

print("Estimated monthly cost: $", round(monthly_cost, 2))


from transformers import pipeline

# Main program
def main():

    print("\n===== HUGGING FACE DEMONSTRATION =====")

    # BERT Fill Mask
    print("\n===== BERT FILL MASK =====")

    # Load BERT model
    bert = pipeline(
        "fill-mask",
        model="google-bert/bert-base-uncased"
    )

    # Give a sentence with a missing word
    sentence = "Artificial intelligence is a [MASK] technology."

    # Predict the missing word
    result = bert(sentence)

    print("\nBERT Predictions:")

    # Display top 3 predictions
    for i in range(3):

        word = result[i]["token_str"]
        score = result[i]["score"]

        print(word, "Score:", round(score, 4))


    # GPT-2 Text Generation
    print("\n===== GPT-2 TEXT GENERATION =====")

    # Load GPT-2 model
    gpt = pipeline(
        "text-generation",
        model="openai-community/gpt2"
    )

    # Give starting text
    prompt = "Artificial intelligence will change the future because"

    # Generate new text
    result = gpt(
        prompt,
        max_new_tokens=40,
        do_sample=False,
        pad_token_id=50256
    )

    print("\nGPT-2 Generated Text:")

    print(result[0]["generated_text"])


# Start the program
main()

def run():
    main()

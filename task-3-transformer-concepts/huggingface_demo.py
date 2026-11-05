
from transformers import pipeline


def run():

    print("\n========== HUGGING FACE DEMONSTRATION ==========")

    # BERT: Fill-Mask task
    print("\n========== BERT FILL-MASK ==========")

    fill_mask = pipeline(
        "fill-mask",
        model="google-bert/bert-base-uncased"
    )

    bert_result = fill_mask(
        "Artificial intelligence is a [MASK] technology."
    )

    print("\nBERT Predictions:")

    for result in bert_result[:3]:
        print(
            result["token_str"],
            "Score:",
            round(result["score"], 4)
        )

    # GPT-2: Text Generation
    print("\n========== GPT-2 TEXT GENERATION ==========")

    generator = pipeline(
        "text-generation",
        model="openai-community/gpt2"
    )

    prompt = "Artificial intelligence will change the future because"

    gpt_result = generator(
        prompt,
        max_new_tokens=40,
        do_sample=False,
        pad_token_id=50256
    )

    print("\nGPT-2 Generated Text:")

    print(gpt_result[0]["generated_text"])


if __name__ == "__main__":
    run()

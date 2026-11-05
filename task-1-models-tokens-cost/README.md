
# Task 1 - Models, Tokens and Cost

## Objective
To understand foundation models, tokenization, context windows and API pricing.

## Technologies Used
- Python
- VS Code
- tiktoken
- Git
- GitHub

## 1. Model Comparison

| Model | Owner | Weights | Context Window | Strengths |
|---|---|---|---|---|
| GPT | OpenAI | Closed | GPT-5: 400K | Reasoning, coding and tool use |
| Claude | Anthropic | Closed | Up to 1M for supported models | Writing, coding and long documents |
| Llama | Meta | Open weights | Llama 4 Scout: 10M | Local deployment and customization |
| Gemini | Google | Closed | Gemini 2.5 Flash: 1M | Multimodal and long context |

Context windows depend on individual model versions.

## 2. Token Counting

Used the Python tiktoken library with cl100k_base encoding.

Five text types were tested:

1. Sentence
2. Paragraph
3. Page
4. JSON
5. Hindi sentence

Words per token = Total words / Total tokens.

Note: The tokenizer is GPT-compatible. Other model families may produce different token counts.

## 3. Cost Calculation

Chats per day = 1000

Input tokens per chat = 1500

Output tokens per chat = 400

Days per month = 30

Monthly input tokens = 45,000,000

Monthly output tokens = 12,000,000

Input price = $0.30 per million tokens

Output price = $2.50 per million tokens

Monthly estimated cost = $43.50

## 4. Context Window Comparison

Compared the five text samples against:

- 8K context window = 8192 tokens
- 128K context window = 131072 tokens

All five sample texts fit within both context windows.

## 5. Model Recommendations

### Public Customer-Support Bot: Gemini Flash

1. Fast responses.
2. Large context window.
3. Multimodal capabilities.
4. API integration.
5. Suitable for hosted customer-support applications.

### Private On-Premise Tool: Meta Llama

1. Downloadable model weights.
2. Local deployment.
3. Better infrastructure control.
4. Customization options.
5. Reduced dependence on external APIs.

## 6. Conclusion

Foundation models are trained on large datasets and can perform different language tasks. Tokens are the basic units processed by language models. Context windows determine how much information a model can handle. API pricing helps estimate the operating cost of AI applications.

## 7. Official References

- OpenAI: https://developers.openai.com/api/docs/models/gpt-5
- Anthropic: https://platform.claude.com/docs/en/models/overview
- Meta: https://www.llama.com/docs/
- Google: https://ai.google.dev/gemini-api/docs/models
- Gemini Pricing: https://ai.google.dev/gemini-api/docs/pricing
- Tiktoken: https://github.com/openai/tiktoken

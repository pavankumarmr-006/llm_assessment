
# Task 3 - Transformer Concepts in Code

## Objective

To understand and implement transformer concepts using Python, NumPy and Hugging Face.

## Technologies Used

- Python
- NumPy
- Matplotlib
- Hugging Face Transformers
- PyTorch
- VS Code

## 1. Single-Head Self-Attention

Implemented Query (Q), Key (K) and Value (V) matrices.

Calculated attention scores:

Attention Scores = Q @ K.T / sqrt(d)

Applied row-wise softmax to obtain attention weights.

Calculated final attention output:

Output = Attention Weights @ V

Verified that every row of the 4x4 attention matrix sums to 1.0.

## 2. Multi-Head Attention

Implemented two independent attention heads.

Concatenated both head outputs and applied an output projection matrix.

## 3. Positional Encoding

Implemented sinusoidal positional encoding using sine and cosine functions.

Positions = 10

Dimensions = 8

Generated and saved a positional encoding heatmap:

output/positional_encoding.png

## 4. Hugging Face Demonstration

### BERT Fill-Mask

Used the pretrained BERT model to predict a missing word in a sentence.

Model: google-bert/bert-base-uncased

Task: Fill-mask

### GPT-2 Text Generation

Used the pretrained GPT-2 model to generate text from an input prompt.

Model: openai-community/gpt2

Task: Text generation

## 5. Encoder-only vs Decoder-only

1. Encoder-only models such as BERT understand input text using bidirectional context.
2. Decoder-only models such as GPT-2 generate text by predicting the next token.
3. BERT is commonly used for classification, understanding and fill-mask tasks.
4. GPT-2 is commonly used for text generation and continuation.
5. Transformers can process tokens in parallel during training, unlike sequential RNN processing.
6. Self-attention helps transformers capture long-range relationships more effectively than traditional RNNs.

## Conclusion

This project demonstrates single-head attention, multi-head attention and sinusoidal positional encoding.

It also demonstrates BERT fill-mask and GPT-2 text generation using Hugging Face.

The experiments help explain how transformer models process and generate language.

## References

- Attention Is All You Need:
  https://arxiv.org/abs/1706.03762

- NumPy:
  https://numpy.org/doc/

- Matplotlib:
  https://matplotlib.org/stable/

- Hugging Face Transformers:
  https://huggingface.co/docs/transformers/

- BERT:
  https://huggingface.co/google-bert/bert-base-uncased

- GPT-2:
  https://huggingface.co/openai-community/gpt2

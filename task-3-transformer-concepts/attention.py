
import numpy as np


def softmax(x):
    """Calculate stable row-wise softmax."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def single_head_attention(X):
    print("\n========== SINGLE-HEAD SELF-ATTENTION ==========")

    np.random.seed(42)

    # Four tokens, each represented by four features
    Wq = np.random.randn(4, 4)
    Wk = np.random.randn(4, 4)
    Wv = np.random.randn(4, 4)

    # Query, Key and Value
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    # Attention scores
    d = Q.shape[1]
    scores = (Q @ K.T) / np.sqrt(d)

    # Row-wise softmax
    attention_weights = softmax(scores)

    # Attention output
    output = attention_weights @ V

    print("\nQuery (Q):\n", Q)
    print("\nKey (K):\n", K)
    print("\nValue (V):\n", V)

    print("\nAttention Weights (4x4):")
    print(np.round(attention_weights, 4))

    print("\nRow sums:")
    print(np.sum(attention_weights, axis=1))

    print("\nAttention Output:")
    print(np.round(output, 4))

    # Verify every row sums to 1
    assert np.allclose(
        np.sum(attention_weights, axis=1),
        1.0
    )

    print("\nVerification: Every row sums to 1.0")


def multi_head_attention(X):
    print("\n========== TWO-HEAD SELF-ATTENTION ==========")

    np.random.seed(10)

    input_dim = X.shape[1]
    num_heads = 2
    head_dim = input_dim // num_heads

    head_outputs = []
    head_weights = []

    for head in range(num_heads):

        # Each head has its own projections
        Wq = np.random.randn(input_dim, head_dim)
        Wk = np.random.randn(input_dim, head_dim)
        Wv = np.random.randn(input_dim, head_dim)

        Q = X @ Wq
        K = X @ Wk
        V = X @ Wv

        scores = (Q @ K.T) / np.sqrt(head_dim)

        weights = softmax(scores)

        head_output = weights @ V

        head_outputs.append(head_output)
        head_weights.append(weights)

        print(f"\nHead {head + 1} Attention Matrix:")
        print(np.round(weights, 4))

        print("Row sums:", np.sum(weights, axis=1))

        assert np.allclose(
            np.sum(weights, axis=1),
            1.0
        )

    # Concatenate both attention heads
    concatenated = np.concatenate(head_outputs, axis=1)

    print("\nConcatenated Output:")
    print(np.round(concatenated, 4))

    # Output projection
    Wo = np.random.randn(input_dim, input_dim)

    final_output = concatenated @ Wo

    print("\nOutput Projection Matrix:")
    print(np.round(Wo, 4))

    print("\nFinal Multi-Head Output:")
    print(np.round(final_output, 4))

    print("\nMulti-head attention completed successfully.")


def run():
    # Four sample token embeddings
    X = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0, 2.0],
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 0.0, 2.0, 0.0]
    ])

    single_head_attention(X)
    multi_head_attention(X)


if __name__ == "__main__":
    run()


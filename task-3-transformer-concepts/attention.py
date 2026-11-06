
import numpy as np

# Function to calculate softmax
def softmax(x):

    # Find exponential values
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))

    # Calculate sum of each row
    total = np.sum(exp_values, axis=1, keepdims=True)

    # Divide to get attention weights
    result = exp_values / total

    return result


# Single Head Attention
def single_head_attention(X):

    print("\n===== SINGLE HEAD ATTENTION =====")

    np.random.seed(42)

    # Create Query, Key and Value weight matrices
    Wq = np.random.randn(4, 4)
    Wk = np.random.randn(4, 4)
    Wv = np.random.randn(4, 4)

    # Calculate Query, Key and Value
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    print("\nQuery:")
    print(Q)

    print("\nKey:")
    print(K)

    print("\nValue:")
    print(V)

    # Calculate attention scores
    scores = (Q @ K.T) / np.sqrt(4)

    print("\nAttention Scores:")
    print(scores)

    # Calculate attention weights using softmax
    weights = softmax(scores)

    print("\nAttention Weights:")
    print(np.round(weights, 4))

    # Calculate final attention output
    output = weights @ V

    print("\nAttention Output:")
    print(np.round(output, 4))

    # Check whether each row adds up to 1
    print("\nRow Sums:")
    print(np.sum(weights, axis=1))

    print("\nSingle Head Attention Completed!")


# Multi Head Attention
def multi_head_attention(X):

    print("\n===== MULTI HEAD ATTENTION =====")

    np.random.seed(10)

    # Number of attention heads
    heads = 2

    # Input size
    size = X.shape[1]

    # Size of each head
    head_size = size // heads

    # Store outputs of both heads
    outputs = []

    # Run two attention heads
    for i in range(heads):

        print("\nHead", i + 1)

        # Create weight matrices
        Wq = np.random.randn(size, head_size)
        Wk = np.random.randn(size, head_size)
        Wv = np.random.randn(size, head_size)

        # Calculate Query, Key and Value
        Q = X @ Wq
        K = X @ Wk
        V = X @ Wv

        # Calculate attention scores
        scores = (Q @ K.T) / np.sqrt(head_size)

        # Calculate attention weights
        weights = softmax(scores)

        print("\nAttention Weights:")
        print(np.round(weights, 4))

        # Calculate attention output
        output = weights @ V

        print("\nHead Output:")
        print(np.round(output, 4))

        # Store the output
        outputs.append(output)

        # Check row sums
        print("\nRow Sums:")
        print(np.sum(weights, axis=1))

    # Join the outputs of both heads
    combined = np.concatenate(outputs, axis=1)

    print("\nCombined Output:")
    print(np.round(combined, 4))

    # Create output weight matrix
    Wo = np.random.randn(size, size)

    # Calculate final output
    final_output = combined @ Wo

    print("\nFinal Multi Head Output:")
    print(np.round(final_output, 4))

    print("\nMulti Head Attention Completed!")


# Main program
def main():

    # Create four sample input tokens
    X = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0, 2.0],
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 0.0, 2.0, 0.0]
    ])

    print("Input Data:")
    print(X)

    # Call single head attention
    single_head_attention(X)

    # Call multi head attention
    multi_head_attention(X)


# Start the program
main()

def run():
    print("\n===== ATTENTION MODULE COMPLETED =====")


import numpy as np
import matplotlib.pyplot as plt


# Function to calculate positional encoding
def positional_encoding(position, d_model):

    # Create an empty matrix
    encoding = np.zeros((position, d_model))

    # Calculate values for each position
    for pos in range(position):

        for i in range(d_model):

            # Calculate angle
            angle = pos / (10000 ** (2 * (i // 2) / d_model))

            # Use sine for even columns
            if i % 2 == 0:
                encoding[pos, i] = np.sin(angle)

            # Use cosine for odd columns
            else:
                encoding[pos, i] = np.cos(angle)

    return encoding


# Main program
def run():

    print("\n===== POSITIONAL ENCODING =====")

    # Number of positions and dimensions
    position = 10
    d_model = 8

    # Calculate positional encoding
    result = positional_encoding(position, d_model)

    # Display the encoding matrix
    print("\nPositional Encoding Matrix:")
    print(np.round(result, 4))

    # Draw the graph
    plt.figure(figsize=(10, 6))

    plt.imshow(result, cmap="viridis", aspect="auto")

    # Add color bar
    plt.colorbar(label="Encoding Value")

    # Add labels
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Position")

    # Add graph title
    plt.title("Sinusoidal Positional Encoding")

    # Show position numbers
    plt.xticks(range(d_model))
    plt.yticks(range(position))

    # Adjust graph layout
    plt.tight_layout()

    # Save the graph
    plt.savefig("output/positional_encoding.png")

    print("\nGraph saved successfully!")

    # Display graph
    plt.show()


# Start the program
run()

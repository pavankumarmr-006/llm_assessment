
import numpy as np
import matplotlib.pyplot as plt


def positional_encoding(position, d_model):

    encoding = np.zeros((position, d_model))

    for pos in range(position):

        for i in range(d_model):

            angle = pos / (
                10000 ** ((2 * (i // 2)) / d_model)
            )

            if i % 2 == 0:
                encoding[pos, i] = np.sin(angle)
            else:
                encoding[pos, i] = np.cos(angle)

    return encoding


def run():

    print("\n========== POSITIONAL ENCODING ==========")

    # 10 positions and 8 dimensions
    position = 10
    d_model = 8

    encoding = positional_encoding(position, d_model)

    print("\nPositional Encoding Matrix (10x8):")
    print(np.round(encoding, 4))

    # Plot the positional encoding
    plt.figure(figsize=(10, 6))

    plt.imshow(
        encoding,
        cmap="viridis",
        aspect="auto"
    )

    plt.colorbar(label="Encoding Value")

    plt.xlabel("Embedding Dimension")
    plt.ylabel("Position")

    plt.title("Sinusoidal Positional Encoding")

    plt.xticks(range(d_model))
    plt.yticks(range(position))

    plt.tight_layout()

    plt.savefig(
        "output/positional_encoding.png",
        dpi=150
    )

    print("\nGraph saved to output/positional_encoding.png")

    plt.show()


if __name__ == "__main__":
    run()

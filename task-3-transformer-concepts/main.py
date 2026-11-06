
# Import programs from other Python files

from attention import run as attention
from positional_encoding import run as positional
from huggingface_demo import run as huggingface


# Main program
def main():

    print("\n==============================")
    print("TASK 3 - TRANSFORMER CONCEPTS")
    print("==============================")

    # Experiment 1 and 2: Attention
    print("\nStarting Attention Experiment...")
    attention()

    # Experiment 3: Positional Encoding
    print("\nStarting Positional Encoding Experiment...")
    positional()

    # Experiment 4: Hugging Face
    print("\nStarting Hugging Face Experiment...")
    huggingface()

    # Display completion message
    print("\n==============================")
    print("ALL EXPERIMENTS COMPLETED")
    print("==============================")


# Start the program
main()

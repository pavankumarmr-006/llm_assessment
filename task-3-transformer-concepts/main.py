
from attention import run as attention_run
from positional_encoding import run as positional_run
from huggingface_demo import run as huggingface_run


def main():

    print("===================================")
    print("TASK 3 - TRANSFORMER CONCEPTS")
    print("===================================")

    # Experiment 1 and 2
    attention_run()

    # Experiment 3
    positional_run()

    # Experiment 4
    huggingface_run()

    print("\n===================================")
    print("ALL EXPERIMENTS COMPLETED")
    print("===================================")


if __name__ == "__main__":
    main()

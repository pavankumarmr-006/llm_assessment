
import json
import re
import requests
from pathlib import Path
from pydantic import BaseModel, ValidationError


# Model details
MODEL = "llama3.1:8b"
API_URL = "http://localhost:11434/api/chat"

# Create folders
BASE_DIR = Path(__file__).parent
RESULT_DIR = BASE_DIR / "results"
PROMPT_DIR = BASE_DIR / "prompts"

RESULT_DIR.mkdir(exist_ok=True)


# Function to ask questions to Ollama
def ask_model(prompt, temperature=0, top_p=0.95, json_mode=False):

    data = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": 200
        }
    }

    # Enable JSON output if needed
    if json_mode:
        data["format"] = "json"

    response = requests.post(API_URL, json=data, timeout=600)

    response.raise_for_status()

    result = response.json()

    return result["message"]["content"]


# Function to convert text into JSON
def parse_json(text):

    text = text.strip()

    # Remove unnecessary code marks
    text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE)

    try:
        result = json.loads(text)
        return result

    except json.JSONDecodeError:
        return None


# Function to save results in a JSON file
def save_json(filename, data):

    file_path = RESULT_DIR / filename

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    print("\nFile saved:", filename)


# =====================================
# EXPERIMENT 1: CLASSIFICATION
# =====================================

def classification_test():

    print("\n===== CLASSIFICATION EXPERIMENT =====")

    # Read ticket data
    with open(BASE_DIR / "golden_set.json", encoding="utf-8") as file:
        tickets = json.load(file)

    versions = ["v1", "v2", "v3"]

    results = {}

    # Test three different prompts
    for version in versions:

        print("\nTesting Prompt:", version)

        # Read prompt file
        file_path = PROMPT_DIR / ("prompt_" + version + ".txt")

        prompt_text = file_path.read_text(encoding="utf-8")

        correct = 0
        details = []

        # Check every ticket
        for ticket in tickets:

            prompt = prompt_text.replace(
                "{ticket}", ticket["ticket"]
            )

            try:

                answer = ask_model(prompt, json_mode=True)

                data = parse_json(answer)

                category = ""
                priority = ""

                if isinstance(data, dict):
                    category = data.get("category", "")
                    priority = data.get("priority", "")

                # Compare predicted and correct answers
                if category == ticket["category"] and priority == ticket["priority"]:
                    correct = correct + 1
                    status = True
                else:
                    status = False

                details.append({
                    "id": ticket["id"],
                    "ticket": ticket["ticket"],
                    "expected_category": ticket["category"],
                    "expected_priority": ticket["priority"],
                    "predicted_category": category,
                    "predicted_priority": priority,
                    "correct": status
                })

            except Exception as error:

                print("Error:", error)

                details.append({
                    "id": ticket["id"],
                    "correct": False,
                    "error": str(error)
                })

        # Calculate accuracy
        accuracy = (correct / len(tickets)) * 100

        print("Correct Answers:", correct)
        print("Total Tickets:", len(tickets))
        print("Accuracy:", round(accuracy, 2), "%")

        results[version] = {
            "correct": correct,
            "total": len(tickets),
            "accuracy": accuracy,
            "details": details
        }

    save_json("classification_results.json", results)


# =====================================
# EXPERIMENT 2: REASONING
# =====================================

def reasoning_test():

    print("\n===== REASONING EXPERIMENT =====")

    # Sample mathematical questions
    problems = [
        {
            "question": "A shop sells 4 pens for Rs. 10 each. What is the total cost?",
            "answer": "40"
        },
        {
            "question": "A car travels 60 km per hour for 3 hours. How far does it travel?",
            "answer": "180"
        },
        {
            "question": "There are 24 apples shared equally among 6 people. How many does each person get?",
            "answer": "4"
        },
        {
            "question": "A shirt costs Rs. 800 and has a 25% discount. What is the final price?",
            "answer": "600"
        },
        {
            "question": "A train travels 120 km in 2 hours. What is its average speed in km per hour?",
            "answer": "60"
        }
    ]

    results = []

    # Run each problem
    for i in range(len(problems)):

        question = problems[i]["question"]
        correct_answer = problems[i]["answer"]

        print("\nProblem", i + 1)
        print(question)

        # Test two different prompting methods
        for mode in ["Plain", "Step by step"]:

            if mode == "Plain":

                prompt = question + " Give the final numerical answer."

            else:

                prompt = question + " Show the calculation and give the final answer."

            try:

                answer = ask_model(prompt)

                print(mode, "Answer:", answer)

                results.append({
                    "problem": i + 1,
                    "mode": mode,
                    "question": question,
                    "expected_answer": correct_answer,
                    "model_answer": answer
                })

            except Exception as error:

                print("Error:", error)

    save_json("reasoning_results.json", results)


# =====================================
# EXPERIMENT 3: CREATIVITY
# =====================================

def creativity_test():

    print("\n===== CREATIVITY EXPERIMENT =====")

    prompt = "Write a short creative story in 4 sentences about a robot helping a village during a storm."

    # Different temperature values
    temperatures = [0, 0.7, 1.2]

    # Different top-p values
    top_p_values = [0.3, 0.95]

    results = []

    # Test different temperature values
    for temp in temperatures:

        for top_p in top_p_values:

            print("\nTemperature:", temp)
            print("Top-p:", top_p)

            # Generate five answers
            for i in range(5):

                try:

                    answer = ask_model(
                        prompt,
                        temperature=temp,
                        top_p=top_p
                    )

                    print("\nRun", i + 1)
                    print(answer)

                    results.append({
                        "temperature": temp,
                        "top_p": top_p,
                        "run": i + 1,
                        "answer": answer
                    })

                except Exception as error:

                    print("Error:", error)

    save_json("creativity_results.json", results)


# =====================================
# EXPERIMENT 4: PYDANTIC MODEL
# =====================================

# Create a model to check resume details
class ResumeData(BaseModel):

    name: str

    skills: list[str]

    years_experience: int


# =====================================
# EXPERIMENT 5: RESUME EXTRACTION
# =====================================

def resume_test():

    print("\n===== RESUME EXTRACTION =====")

    # Read resume data
    with open(BASE_DIR / "resumes.json", encoding="utf-8") as file:
        resumes = json.load(file)

    results = []

    invalid_plain = 0
    invalid_json = 0

    # Check every resume
    for resume in resumes:

        print("\nResume:", resume["id"])

        prompt = """
Extract the following information from this resume:

- name
- skills as a list
- years_experience as an integer

Return valid JSON only.

Resume:
""" + resume["resume"]

        # Test without JSON mode
        try:

            answer1 = ask_model(prompt, json_mode=False)

            data1 = parse_json(answer1)

            try:

                valid1 = ResumeData(**data1)

                plain_valid = True
                plain_output = valid1.model_dump()

            except (ValidationError, TypeError, ValueError):

                plain_valid = False
                plain_output = None
                invalid_plain = invalid_plain + 1

        except Exception as error:

            answer1 = str(error)
            plain_valid = False
            plain_output = None
            invalid_plain = invalid_plain + 1

        # Test with JSON mode
        try:

            answer2 = ask_model(prompt, json_mode=True)

            data2 = parse_json(answer2)

            try:

                valid2 = ResumeData(**data2)

                json_valid = True
                json_output = valid2.model_dump()

            except (ValidationError, TypeError, ValueError):

                json_valid = False
                json_output = None
                invalid_json = invalid_json + 1

        except Exception as error:

            answer2 = str(error)
            json_valid = False
            json_output = None
            invalid_json = invalid_json + 1

        print("Plain JSON Valid:", plain_valid)
        print("JSON Mode Valid:", json_valid)

        results.append({
            "resume_id": resume["id"],
            "expected": resume["expected"],
            "plain_answer": answer1,
            "plain_valid": plain_valid,
            "plain_extracted": plain_output,
            "json_answer": answer2,
            "json_valid": json_valid,
            "json_extracted": json_output
        })

    # Display invalid results
    print("\nInvalid Plain Outputs:", invalid_plain)
    print("Invalid JSON Outputs:", invalid_json)

    save_json("resume_results.json", results)

    summary = {
        "plain_invalid": invalid_plain,
        "json_mode_invalid": invalid_json,
        "total": len(resumes)
    }

    save_json("resume_validation_summary.json", summary)


# =====================================
# EXPERIMENT 6: PROMPT IMPROVEMENT
# =====================================

def prompt_improvement_test():

    print("\n===== PROMPT IMPROVEMENT =====")

    # Bad prompts and their improved versions
    examples = [
        {
            "bad": "Classify this.",

            "improved": "Classify this customer ticket into Billing, Technical or Account. Assign Low, Medium or High priority. Return JSON with category and priority."
        },
        {
            "bad": "Extract resume details.",

            "improved": "Extract the candidate's name, technical skills as a list and years of experience as an integer. Return JSON only."
        },
        {
            "bad": "Write something about AI.",

            "improved": "Write a 150-word beginner-friendly explanation of artificial intelligence. Include its definition, two applications and one limitation."
        }
    ]

    # Display the prompts
    for i in range(len(examples)):

        print("\nExample", i + 1)

        print("Bad Prompt:")
        print(examples[i]["bad"])

        print("\nImproved Prompt:")
        print(examples[i]["improved"])

    # Save prompt examples
    save_json("prompt_improvements.json", examples)


# =====================================
# MAIN PROGRAM
# =====================================

def main():

    print("\n===================================")
    print("TASK 4 - PROMPT ENGINEERING TOOLKIT")
    print("===================================")

    print("\nMake sure Ollama is running.")
    print("Model:", MODEL)

    # Run all experiments
    classification_test()

    reasoning_test()

    creativity_test()

    resume_test()

    prompt_improvement_test()

    print("\n===================================")
    print("ALL EXPERIMENTS COMPLETED")
    print("===================================")

    print("\nCheck the results folder.")


# Start the program
if __name__ == "__main__":
    main()

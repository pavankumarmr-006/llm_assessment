
import json
import csv
import re
import requests
from pathlib import Path
from pydantic import BaseModel, ValidationError


# ==========================================
# CONFIGURATION
# ==========================================

MODEL = "llama3.1:8b"
API_URL = "http://localhost:11434/api/chat"

BASE_DIR = Path(__file__).parent
RESULT_DIR = BASE_DIR / "results"
PROMPT_DIR = BASE_DIR / "prompts"

RESULT_DIR.mkdir(exist_ok=True)


# ==========================================
# OLLAMA API FUNCTION
# ==========================================

def ask_model(prompt, temperature=0, top_p=0.95, json_mode=False):

    payload = {
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

    if json_mode:
        payload["format"] = "json"

    response = requests.post(
        API_URL,
        json=payload,
        timeout=600
    )

    response.raise_for_status()

    return response.json()["message"]["content"]


# ==========================================
# JSON HELPER
# ==========================================

def parse_json(text):

    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)

        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None

    return None


def save_json(filename, data):

    path = RESULT_DIR / filename

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


# ==========================================
# EXPERIMENT 1: CLASSIFICATION
# ==========================================

def classification_test():

    print("\n========== CLASSIFICATION EXPERIMENT ==========")

    with open(
        BASE_DIR / "golden_set.json",
        encoding="utf-8"
    ) as file:
        tickets = json.load(file)

    versions = ["v1", "v2", "v3"]
    all_results = {}

    for version in versions:

        print(f"\nTesting Prompt {version.upper()}")

        prompt_template = (
            PROMPT_DIR / f"prompt_{version}.txt"
        ).read_text(encoding="utf-8")

        correct = 0
        version_results = []

        for item in tickets:

            prompt = prompt_template.replace(
                "{ticket}", item["ticket"]
            )

            try:
                answer = ask_model(
                    prompt,
                    temperature=0,
                    json_mode=True
                )

                data = parse_json(answer)

                predicted_category = (
                    data.get("category", "")
                    if isinstance(data, dict) else ""
                )

                predicted_priority = (
                    data.get("priority", "")
                    if isinstance(data, dict) else ""
                )

                is_correct = (
                    predicted_category == item["category"]
                    and predicted_priority == item["priority"]
                )

                if is_correct:
                    correct += 1

                version_results.append({
                    "id": item["id"],
                    "ticket": item["ticket"],
                    "expected_category": item["category"],
                    "expected_priority": item["priority"],
                    "predicted_category": predicted_category,
                    "predicted_priority": predicted_priority,
                    "correct": is_correct,
                    "raw_answer": answer
                })

            except Exception as error:

                version_results.append({
                    "id": item["id"],
                    "correct": False,
                    "error": str(error)
                })

        accuracy = (correct / len(tickets)) * 100

        print(
            f"{version.upper()}: "
            f"{correct}/15 ({accuracy:.1f}%)"
        )

        all_results[version] = {
            "correct": correct,
            "total": len(tickets),
            "accuracy": accuracy,
            "details": version_results
        }

    save_json("classification_results.json", all_results)


# ==========================================
# EXPERIMENT 2: WORD PROBLEMS
# ==========================================

def reasoning_test():

    print("\n========== REASONING EXPERIMENT ==========")

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

    for index, problem in enumerate(problems, 1):

        print(f"\nProblem {index}")

        for mode in ["Plain", "Step by step"]:

            if mode == "Plain":
                prompt = (
                    problem["question"]
                    + " Give the final numerical answer."
                )
            else:
                prompt = (
                    problem["question"]
                    + " Think step by step and show a brief "
                    "calculation, then give the final answer."
                )

            try:
                answer = ask_model(
                    prompt,
                    temperature=0
                )

                print(mode, ":", answer)

                results.append({
                    "problem": index,
                    "mode": mode,
                    "question": problem["question"],
                    "expected_answer": problem["answer"],
                    "model_answer": answer
                })

            except Exception as error:
                results.append({
                    "problem": index,
                    "mode": mode,
                    "error": str(error)
                })

    save_json("reasoning_results.json", results)


# ==========================================
# EXPERIMENT 3: TEMPERATURE AND TOP-P
# ==========================================

def creativity_test():

    print("\n========== CREATIVITY EXPERIMENT ==========")

    creative_prompt = (
        "Write a short creative story in 4 sentences "
        "about a robot helping a village during a storm."
    )

    temperatures = [0, 0.7, 1.2]
    top_p_values = [0.3, 0.95]

    results = []

    for temperature in temperatures:

        for top_p in top_p_values:

            print(
                f"\nTemperature: {temperature}, "
                f"Top-p: {top_p}"
            )

            for repetition in range(1, 6):

                try:
                    answer = ask_model(
                        creative_prompt,
                        temperature=temperature,
                        top_p=top_p
                    )

                    print(f"Run {repetition}: {answer}")

                    results.append({
                        "temperature": temperature,
                        "top_p": top_p,
                        "run": repetition,
                        "answer": answer
                    })

                except Exception as error:

                    results.append({
                        "temperature": temperature,
                        "top_p": top_p,
                        "run": repetition,
                        "error": str(error)
                    })

    save_json("creativity_results.json", results)


# ==========================================
# EXPERIMENT 4: PYDANTIC MODEL
# ==========================================

class ResumeData(BaseModel):

    name: str

    skills: list[str]

    years_experience: int


# ==========================================
# EXPERIMENT 5: RESUME EXTRACTION
# ==========================================

def resume_test():

    print("\n========== RESUME EXTRACTION ==========")

    with open(
        BASE_DIR / "resumes.json",
        encoding="utf-8"
    ) as file:
        resumes = json.load(file)

    results = []

    invalid_plain = 0
    invalid_json = 0

    for item in resumes:

        print(f"\nResume {item['id']}")

        prompt = f"""
Extract the following information from this resume:

- name
- skills (array of strings)
- years_experience (integer)

Return valid JSON only.

Resume:
{item['resume']}
"""

        # Without JSON mode
        try:

            plain_answer = ask_model(
                prompt,
                temperature=0,
                json_mode=False
            )

            plain_data = parse_json(plain_answer)

            try:
                validated_plain = ResumeData.model_validate(
                    plain_data
                )

                plain_valid = True

                plain_output = validated_plain.model_dump()

            except (ValidationError, TypeError, ValueError):
                plain_valid = False
                plain_output = None
                invalid_plain += 1

        except Exception as error:

            plain_answer = str(error)
            plain_valid = False
            plain_output = None
            invalid_plain += 1

        # With JSON mode
        try:

            json_answer = ask_model(
                prompt,
                temperature=0,
                json_mode=True
            )

            json_data = parse_json(json_answer)

            try:
                validated_json = ResumeData.model_validate(
                    json_data
                )

                json_valid = True

                json_output = validated_json.model_dump()

            except (ValidationError, TypeError, ValueError):
                json_valid = False
                json_output = None
                invalid_json += 1

        except Exception as error:

            json_answer = str(error)
            json_valid = False
            json_output = None
            invalid_json += 1

        print("Plain JSON valid:", plain_valid)
        print("JSON mode valid:", json_valid)

        results.append({
            "resume_id": item["id"],
            "expected": item["expected"],
            "plain_answer": plain_answer,
            "plain_valid": plain_valid,
            "plain_extracted": plain_output,
            "json_answer": json_answer,
            "json_valid": json_valid,
            "json_extracted": json_output
        })

    print("\nInvalid plain outputs:", invalid_plain, "/ 5")
    print("Invalid JSON mode outputs:", invalid_json, "/ 5")

    save_json("resume_results.json", results)

    save_json("resume_validation_summary.json", {
        "plain_invalid": invalid_plain,
        "json_mode_invalid": invalid_json,
        "total": len(resumes)
    })


# ==========================================
# EXPERIMENT 6: BAD PROMPTS AND IMPROVEMENTS
# ==========================================

def prompt_improvement_test():

    print("\n========== PROMPT IMPROVEMENT ==========")

    examples = [
        {
            "bad": "Classify this.",
            "improved": (
                "Classify this customer ticket into Billing, Technical "
                "or Account. Assign Low, Medium or High priority. "
                "Return JSON with category and priority."
            )
        },
        {
            "bad": "Extract resume details.",
            "improved": (
                "Extract the candidate's full name, technical skills "
                "as a string array and total years of experience "
                "as an integer. Return valid JSON only."
            )
        },
        {
            "bad": "Write something about AI.",
            "improved": (
                "Write a 150-word beginner-friendly explanation of "
                "artificial intelligence. Include its definition, "
                "two applications and one limitation. Use simple English."
            )
        }
    ]

    save_json("prompt_improvements.json", examples)

    for index, example in enumerate(examples, 1):

        print(f"\nExample {index}")
        print("Bad prompt:", example["bad"])
        print("Improved prompt:", example["improved"])


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    print("======================================")
    print("TASK 4 - PROMPT ENGINEERING TOOLKIT")
    print("======================================")

    print("\nMake sure Ollama is running.")
    print("Model:", MODEL)

    classification_test()

    reasoning_test()

    creativity_test()

    resume_test()

    prompt_improvement_test()

    print("\n======================================")
    print("ALL EXPERIMENTS COMPLETED")
    print("======================================")

    print("\nCheck the results folder for experiment outputs.")


if __name__ == "__main__":
    main()

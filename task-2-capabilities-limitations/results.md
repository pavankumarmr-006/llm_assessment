
# Task 2 - Capabilities and Limitations Lab

## Results Table

| Test | Prompt | Result | Rating |
|---|---|---|---|
| Summarisation | Summarise a sample business paragraph in 3 bullet points. | To be filled after testing | /5 |
| Classification | Classify a customer review as Positive, Negative or Neutral. | To be filled after testing | /5 |
| Extraction | Extract customer details into JSON format. | To be filled after testing | /5 |
| Translation | Translate an English sentence into Hindi. | To be filled after testing | /5 |
| Code Generation | Generate Python code to calculate sum and average. | To be filled after testing | /5 |
| Knowledge Cutoff | Ask about a very recent event. | To be filled after testing | /5 |
| Hallucination - First Test | Ask about a fictional research paper without warning. | To be filled after testing | /5 |
| Hallucination - Second Test | Ask about the same paper and say I don't know if unsure. | To be filled after testing | /5 |
| Context Length | Find a project code hidden in a long text. | To be filled after testing | /5 |
| Arithmetic | Solve a shop bill problem involving discount and tax. | To be filled after testing | /5 |
| Logic Puzzle | Solve the 100 machines production puzzle. | To be filled after testing | /5 |

## Arithmetic Verification

- Notebook cost: Rs. 120
- Pen cost: Rs. 30
- Original subtotal: Rs. 150
- Discount: Rs. 15
- Price after discount: Rs. 135
- Tax: Rs. 24.30
- Final amount: Rs. 159.30

## Logic Puzzle Verification

Correct answer: 1 minute.

100 machines produce 100 products in 1 minute.

## Five Business Application Safeguards

1. Input validation: Validate user inputs before processing.
2. Human verification: Important AI decisions should be checked by a human.
3. Hallucination prevention: Instruct the model to admit when it does not know something.
4. Data privacy: Protect sensitive customer information.
5. Output validation: Check AI-generated answers before showing them to customers.

## Conclusion

LLMs can perform summarisation, classification, extraction, translation and code generation. However, they can also produce hallucinations, miss recent information, forget details in long contexts and make reasoning mistakes.

Business applications should use proper verification and safety measures.
***********************************************************************
Test 1 – Summarisation
----------------------
propt
-----
Summarise this text in exactly 3 bullet points.

Artificial intelligence is becoming important in modern businesses.
Companies use AI to automate repetitive tasks, improve customer
service, analyse data and support decision-making. AI chatbots
can answer customer questions quickly and reduce employee workload.
However, businesses must consider data privacy, accuracy and security.
AI helps businesses automate repetitive tasks, improve customer service, analyse data and support decision-making.
-----------------------------------------------------------------------
result
------
AI chatbots provide quick answers to customer questions and reduce employees' workload.

Businesses must consider data privacy, accuracy and security when using AI.
***********************************************************************
Test 2 – Classification
-----------------------
Classify this customer review as Positive, Negative or Neutral.

"The delivery was late, the package was damaged,
and customer support did not respond to my complaint."

Give the sentiment and one reason.
-----------------------------------------------------------------------
result
------
Sentiment: Negative

Reason: The customer experienced late delivery, a damaged package, and unresponsive customer support.
***********************************************************************

Test 3 – Extraction
-------------------
text
Extract the following information and return JSON only.

Customer Name: Rahul Sharma
Order ID: ORD2026
Product: Wireless Headphones
Amount: Rs. 2499
Order Date: 15 October 2026
Delivery Status: Shipped

Extract customer name, order ID, product, amount,
order date and delivery status.
Do not invent missing information.
-----------------------------------------------------------------------
result
-------
{
  "customer_name": "Rahul Sharma",
  "order_id": "ORD2026",
  "product": "Wireless Headphones",
  "amount": "Rs. 2499",
  "order_date": "15 October 2026",
  "delivery_status": "Shipped"
}
***********************************************************************
Test 4 – Translation
---------------------

text
Translate this English sentence into Hindi:

Technology helps people communicate,
learn new skills and solve problems.
-----------------------------------------------------------------------
result
-------
प्रौद्योगिकी लोगों को संवाद करने, नए कौशल सीखने और समस्याओं को हल करने में मदद करती है।
***********************************************************************
Test 5 – Code generation
--------------------------
text
Write a Python program that takes a list of numbers
and calculates the sum and average.

Requirements:
1. Use a function.
2. Handle an empty list safely.
3. Include comments.
4. Show an example.
-----------------------------------------------------------------------
result
-------
# Function to calculate sum and average of a list
def calculate_sum_average(numbers):

    # Check if the list is empty
    if not numbers:
        return 0, 0

    # Calculate sum
    total = sum(numbers)

    # Calculate average
    average = total / len(numbers)

    return total, average


# Example list
numbers = [10, 20, 30, 40, 50]

# Call the function
total, average = calculate_sum_average(numbers)

# Display the results
print("Numbers:", numbers)
print("Sum:", total)
print("Average:", average)
-----------------------------------------------------------------------
output
-------
Numbers: [10, 20, 30, 40, 50]
Sum: 150
Average: 30.0
***********************************************************************
Knowledge cutoff test

Prompt:

"What is the latest major development in the world as of November 5, 2026?"

Result:
As of November 5, 2026, one of the biggest international developments is the major political shift in the United States following the 2026 midterm elections.
What happened?

The Democratic Party achieved a major victory in the US midterm elections.

The result gives Democrats an opportunity to gain greater control over Congress.

Democrats have promised to challenge President Donald Trump's policies.

The election result could affect US decisions on international relations, trade, security and economic policy.

Why is it important?

The US plays a major role in global politics and the world economy. A change in congressional power could influence government decisions and America's relationships with other countries.

Date: November 4–5, 2026.

Sources:

Reuters – International reactions to Democrats winning US midterms 

Reuters – Global market reaction to the election 
Rating: 3/5
***********************************************************************

Hallucination test
First prompt:

"Explain the research paper 'Artificial Intelligence and Quantum Learning in Rural Education by Dr. Alex Johnson (2025).'"

Result:
I don't know.

I cannot verify that a research paper titled "Artificial Intelligence and Quantum Learning in Rural Education" by Dr. Alex Johnson (2025) actually exists.

Without a verified paper or its contents, I should not invent its research findings, methodology, or conclusions.


Rating: 2/5
***********************************************************************5. Context length test
Prompt:

"Read the following text and identify the special code mentioned in the middle."

Sample text:


text
A company was established in 2010.
It provides software services.
It has offices in multiple cities.
Its employees work on different projects.
The company's internal project code is BLUE742.
It serves customers from different industries.
It develops web and mobile applications.
It focuses on customer satisfaction.
The company plans to expand its services.
Result:

The LLM correctly identified the special code as BLUE742.

Observation: The model can retrieve information from the middle of a text. However, performance may decrease when the context becomes extremely long or contains distracting information.

Rating: 5/5


from huggingface_hub import InferenceClient
from datasets import load_dataset
import pandas as pd
from scipy.stats import pearsonr

# Load the feedbackQA dataset
dataset = load_dataset("McGill-NLP/feedbackQA", trust_remote_code=True )

# Initialize the model client
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
client = InferenceClient(model=model_name)

# Example function to evaluate answers
def judge_answer(question, answer):
    prompt = f"""
    Evaluate the following answer based on a scale of 1 to 4. The scale means:
    4: Excellent answer.
    3: Good answer.
    2: Fair answer.
    1: Poor answer.

    Question: {question}
    Answer: {answer}
    Score:
    """
    response = client.text(prompt)
    return int(response.strip().split()[-1])  # Extract the score from the output

# Test on a small subset
questions = dataset['train']['question'][:10]
answers = dataset['train']['answer'][:10]

scores = []
for question, answer in zip(questions, answers):
    score = judge_answer(question, answer)
    scores.append(score)

# Print results
df = pd.DataFrame({"Question": questions, "Answer": answers, "Score": scores})
print(df)

# Optional: Calculate correlation with human scores
human_scores = dataset['train']['human_scores'][:10]
correlation, _ = pearsonr(scores, human_scores)
print(f"Pearson Correlation with human scores: {correlation}")

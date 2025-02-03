
"""
This file contains the different evaluation metrics used to measure the output of the different models
"""

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score


"""
Exact Match: Returns 1.0 if the prediction exactly matches the ground truth; otherwise, 0.0.
"""
def exact_match(prediction, ground_truth):
    return 1.0 if prediction == ground_truth else 0.0


"""
BLEU Score: Measures the n-gram overlap between the prediction and ground truth. A higher score indicates better quality.
"""
def calculate_bleu(prediction, ground_truth):
    reference = [ground_truth.split()]
    candidate = prediction.split()
    smoothing_function = SmoothingFunction().method1
    return sentence_bleu(reference, candidate, smoothing_function=smoothing_function)

"""
ROUGE Score: Evaluates the overlap of n-grams between the prediction and ground truth, focusing on recall.
"""
def calculate_rouge(prediction, ground_truth):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth, prediction)
    return scores['rouge1'].fmeasure, scores['rougeL'].fmeasure

"""
F1-Score: The harmonic mean of precision and recall, providing a balance between the two.
"""
def calculate_f1(prediction, ground_truth):
    prediction_tokens = set(prediction.split())
    ground_truth_tokens = set(ground_truth.split())
    intersection = prediction_tokens.intersection(ground_truth_tokens)
    precision = len(intersection) / len(prediction_tokens) if prediction_tokens else 0
    recall = len(intersection) / len(ground_truth_tokens) if ground_truth_tokens else 0
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

"""
Partial Match: Measures the proportion of tokens in the ground truth that are also in the prediction.
"""
def partial_match(prediction, ground_truth):
    prediction_tokens = set(prediction.split())
    ground_truth_tokens = set(ground_truth.split())
    intersection = prediction_tokens.intersection(ground_truth_tokens)
    return len(intersection) / len(ground_truth_tokens) if ground_truth_tokens else 0


# Example texts
prediction = "The quick brown fox jumps over the lazy dog."
ground_truth = "The quick brown fox jumped over the lazy dog."

# Exact Match
em_score = exact_match(prediction, ground_truth)
print(f"Exact Match: {em_score}")

# BLEU Score
bleu_score = calculate_bleu(prediction, ground_truth)
print(f"BLEU Score: {bleu_score:.2f}")

# ROUGE Score
rouge1, rougeL = calculate_rouge(prediction, ground_truth)
print(f"ROUGE-1: {rouge1:.2f}")
print(f"ROUGE-L: {rougeL:.2f}")

# F1-Score
f1 = calculate_f1(prediction, ground_truth)
print(f"F1-Score: {f1:.2f}")

# Partial Match
partial = partial_match(prediction, ground_truth)
print(f"Partial Match: {partial:.2f}")

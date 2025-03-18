
"""
This file contains the different evaluation metrics used to measure the output of the different models
"""

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score

import pdb

class MeasureClass:
    def __init__(self):
        self.prediction = None
        self.gound_truth = None

    """
    Exact Match: Returns 1.0 if the prediction exactly matches the ground truth; otherwise, 0.0.
    """
    def exact_match(self):
        return 1.0 if self.prediction == self.ground_truth else 0.0
    
    
    """
    BLEU Score: Measures the n-gram overlap between the prediction and ground truth. A higher score indicates better quality.
    """
    def calculate_bleu(self):
        reference = [self.ground_truth.split()]
        candidate = self.prediction.split()
        smoothing_function = SmoothingFunction().method1
        return sentence_bleu(reference, candidate, smoothing_function=smoothing_function)
    
    """
    ROUGE Score: Evaluates the overlap of n-grams between the prediction and ground truth, focusing on recall.
    """
    def calculate_rouge1(self):
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        scores = scorer.score(self.ground_truth, self.prediction)
        return scores['rouge1'].fmeasure
        
        
    """   
    ROUGE Score: Evaluates the overlap of n-grams between the prediction and ground truth, focusing on recall.
    """
    def calculate_rougeL(self):
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(self.ground_truth, self.prediction)
        return scores['rougeL'].fmeasure
        
    """
    F1-Score: The harmonic mean of precision and recall, providing a balance between the two.
    """
    def calculate_f1(self):
        prediction_tokens = set(self.prediction.split())
        ground_truth_tokens = set(self.ground_truth.split())
        intersection = prediction_tokens.intersection(ground_truth_tokens)
        precision = len(intersection) / len(prediction_tokens) if prediction_tokens else 0
        recall = len(intersection) / len(ground_truth_tokens) if ground_truth_tokens else 0
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    """
    Partial Match: Measures the proportion of tokens in the ground truth that are also in the prediction.
    """
    def partial_match(self):
        prediction_tokens = set(self.prediction.split())
        ground_truth_tokens = set(self.ground_truth.split())
        intersection = prediction_tokens.intersection(ground_truth_tokens)
        return len(intersection) / len(ground_truth_tokens) if ground_truth_tokens else 0

    """
    This function compares two strings with respect to all measuers.
    """
    def compare_using_all_measures(self):
        scores = [];
        scores.append(self.exact_match())
        scores.append(self.calculate_bleu())
        scores.append(self.calculate_rouge1())
        scores.append(self.calculate_rougeL())
        scores.append(self.calculate_f1())
        scores.append(self.partial_match())

        return scores

    """
    This function print the resutls of comparing two measures neatly.
    """
    def neatly_print_measures(self):
        pdb.set_trace()
        print(f"Prediction : {self.prediction}\n Ground Truth: {self.ground_truth}\n Exact Match: {self.exact_match()} \n blue_score: {self.calculate_bleu()}\n rouge1_score:{self.calculate_rouge1()}\n rougeL_score:{self.calculate_rougeL()}\n f1_score: {self.calculate_f1()}\n partial matching: {self.partial_match()}")

        
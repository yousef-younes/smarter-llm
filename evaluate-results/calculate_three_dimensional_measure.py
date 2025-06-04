from typing import List, Dict, Tuple
from scipy.stats import kendalltau
import numpy as np


def normalize(value):
    """Normalize a field value for comparison."""
    if value is None:
        return "none"
    return str(value).strip().lower()


def get_field_order_list(authors: List[Dict], field: str) -> List[str]:
    """Return the ordered list of field values."""
    return [normalize(author.get(field)) for author in authors]


def compute_kendall_tau_order_accuracy(actual: List[str], pred: List[str]) -> float:
    """Compute Kendall Tau correlation between actual and predicted field orders."""
    # Only consider values present in both lists
    common_values = set(actual) & set(pred)

    if len(common_values) < 2:
        return 1.0  # Not enough data to evaluate order

    # Map field value to position
    actual_pos = {val: idx for idx, val in enumerate(actual) if val in common_values}
    pred_pos = {val: idx for idx, val in enumerate(pred) if val in common_values}

    # Get orderings
    actual_ranks = [actual_pos[val] for val in common_values]
    pred_ranks = [pred_pos[val] for val in common_values]

    tau, _ = kendalltau(actual_ranks, pred_ranks)
    return max(0.0, tau if tau is not None else 0.0)


def evaluate_metadata_fields(actual: List[Dict], pred: List[Dict], fields: List[str]) -> Tuple[int, int, int, int]:
    tp = fp = fn = tn = 0
    min_len = min(len(actual), len(pred))

    for i in range(min_len):
        for field in fields:
            a_val = normalize(actual[i].get(field))
            p_val = normalize(pred[i].get(field))

            if a_val == p_val:
                tp += 1
            elif a_val == "none" and p_val == "NA":
                tn += 1
            elif a_val == "none" and p_val != "NA":
                fp += 1
            elif a_val != "none" and p_val == "NA":
                fn += 1

    return tp, fp, fn, tn


def compute_composite_score(
    authors_actual: List[Dict],
    authors_pred: List[Dict],
    weights: Tuple[float, float, float] = (0.2, 0.4, 0.4)
) -> Dict:
    fields = ['name', 'email', 'affiliation']

    # 1. Author Count Accuracy (ACA)
    n_actual = len(authors_actual)
    n_pred = len(authors_pred)
    n = max(n_actual, n_pred)
    aca = 1 - abs(n_actual - n_pred) / n if n > 0 else 1.0

    # 2. Author Order Accuracy (average across fields)
    aoa_fields = []
    for field in fields:
        actual_order = get_field_order_list(authors_actual, field)
        pred_order = get_field_order_list(authors_pred, field)
        tau = compute_kendall_tau_order_accuracy(actual_order, pred_order)
        aoa_fields.append(tau)
    aoa = sum(aoa_fields) / len(aoa_fields)

    # 3. Metadata Field Evaluation (TP, FP, FN, TN)
    tp, fp, fn, tn = evaluate_metadata_fields(authors_actual, authors_pred, fields)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # 4. Composite Score
    w1, w2, w3 = weights
    total_score = w1 * aca + w2 * aoa + w3 * f1

    return {
        'ACA': round(aca, 4),
        'AOA': round(aoa, 4),
        'F1-Score': round(f1, 4),
        'Composite-core': round(total_score, 4),
        'Details': {
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'Precision': round(precision, 4),
            'Recall': round(recall, 4)
        }
    }

import string
import re
from collections import Counter

from sklearn.metrics import f1_score as skf1

def evaluate(eval_file, answer_dict, version):
    if version == "v2.0":
        f1 = exact_match = total = correct_answerable = 0
        z_true, z_pred = [], []
        for key, value in answer_dict.items():
            total += 1
            is_answerable = len(eval_file[key]["answers"]) > 0
            ground_truths = eval_file[key]["answers"] if is_answerable else [""]
            prediction = value
            if (prediction != "" and is_answerable) or (
                prediction == "" and not is_answerable
            ):
                correct_answerable += 1

            z_true.append(1 if is_answerable else 0)
            z_pred.append(1 if prediction == "" else 0)
            

            exact_match += metric_max_over_ground_truths(
                exact_match_score, prediction, ground_truths
            )
            f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total
        answerability_acc = 100.0 * correct_answerable / total
        return {
            "exact_match": exact_match,
            "f1": f1,
            "answerability_acc": answerability_acc,
            "answerability_f1": skf1(z_true, z_pred)
        }
    else:
        f1 = exact_match = total = 0
        for key, value in answer_dict.items():
            total += 1
            ground_truths = eval_file[key]["answers"]
            prediction = value
            exact_match += metric_max_over_ground_truths(
                exact_match_score, prediction, ground_truths
            )
            f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total
        return {"exact_match": exact_match, "f1": f1}


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        # exclude.update('，', '。', '、', '；', '「', '」')
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if len(ground_truth_tokens) == 0 or len(prediction_tokens) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(ground_truth_tokens == prediction_tokens)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

import re
from collections import Counter

import torch
import numpy as np
from torch.utils.data import Dataset

from config import config


class SQuADDataset(Dataset):
    def __init__(self, npz_file, batch_size):
        data = np.load(npz_file)
        self.context_idxs = data["context_idxs"]
        self.context_char_idxs = data["context_char_idxs"]
        self.ques_idxs = data["ques_idxs"]
        self.ques_char_idxs = data["ques_char_idxs"]
        self.y1s = data["y1s"]
        self.y2s = data["y2s"]
        self.ids = data["ids"]
        if config.data_version == "V2":
            self.impossibles = data["impossibles"].astype(np.int)
        self.num = len(self.ids)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if config.data_version == "V2":
            return (
                self.context_idxs[idx],
                self.context_char_idxs[idx],
                self.ques_idxs[idx],
                self.ques_char_idxs[idx],
                self.y1s[idx],
                self.y2s[idx],
                self.ids[idx],
                self.impossibles[idx],
            )
        else:
            return (
                self.context_idxs[idx],
                self.context_char_idxs[idx],
                self.ques_idxs[idx],
                self.ques_char_idxs[idx],
                self.y1s[idx],
                self.y2s[idx],
                self.ids[idx],
            )


class EMA:
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}
        self.original = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, model, num_updates):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                decay = min(self.mu, (1 + num_updates) / (10 + num_updates))
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]


def get_loader(npz_file, batch_size):
    if config.data_version == "V2":

        def collate(data):
            Cwid, Ccid, Qwid, Qcid, y1, y2, ids, impossibles = zip(*data)
            Cwid = torch.tensor(Cwid).long()
            Ccid = torch.tensor(Ccid).long()
            Qwid = torch.tensor(Qwid).long()
            Qcid = torch.tensor(Qcid).long()
            y1 = torch.from_numpy(np.array(y1)).long()
            y2 = torch.from_numpy(np.array(y2)).long()
            ids = torch.from_numpy(np.array(ids)).long()
            impossibles = torch.from_numpy(np.array(impossibles)).long()
            return Cwid, Ccid, Qwid, Qcid, y1, y2, ids, impossibles

    else:

        def collate(data):
            Cwid, Ccid, Qwid, Qcid, y1, y2, ids = zip(*data)
            Cwid = torch.tensor(Cwid).long()
            Ccid = torch.tensor(Ccid).long()
            Qwid = torch.tensor(Qwid).long()
            Qcid = torch.tensor(Qcid).long()
            y1 = torch.from_numpy(np.array(y1)).long()
            y2 = torch.from_numpy(np.array(y2)).long()
            ids = torch.from_numpy(np.array(ids)).long()
            return Cwid, Ccid, Qwid, Qcid, y1, y2, ids

    dataset = SQuADDataset(npz_file, batch_size)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=5,
        collate_fn=collate,
    )
    return data_loader


def convert_tokens(eval_file, qa_id, pp1, pp2, zz=None):

    answer_dict = {}
    remapped_dict = {}
    if config.data_version == "V2":
        for qid, p1, p2, z in zip(qa_id, pp1, pp2, zz):
            if float(z) == 1:
                answer_dict[str(qid)] = ""
                remapped_dict[str(qid)] = ""
            else:
                context = eval_file[str(qid)]["context"]
                spans = eval_file[str(qid)]["spans"]
                uuid = eval_file[str(qid)]["uuid"]
                print(qid)
                start_idx = spans[p1][0]
                end_idx = spans[p2][1]
                answer_dict[str(qid)] = context[start_idx:end_idx]
                remapped_dict[uuid] = context[start_idx:end_idx]
    else:
        for qid, p1, p2 in zip(qa_id, pp1, pp2):
            context = eval_file[str(qid)]["context"]
            spans = eval_file[str(qid)]["spans"]
            uuid = eval_file[str(qid)]["uuid"]
            start_idx = spans[p1][0]
            end_idx = spans[p2][1]
            answer_dict[str(qid)] = context[start_idx:end_idx]
            remapped_dict[uuid] = context[start_idx:end_idx]
    return answer_dict, remapped_dict


def evaluate(eval_file, answer_dict):
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

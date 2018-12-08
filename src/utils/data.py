import torch
from torch.utils.data import Dataset
import numpy as np


class SQuADDataset(Dataset):
    def __init__(self, npz_file, batch_size, version):
        data = np.load(npz_file)
        self.version = version

        idx = np.arange(len(data["y1s"]))
        # np.random.seed(0)
        # np.random.shuffle(idx)

        self.context_idxs = data["context_idxs"][idx]
        self.context_char_idxs = data["context_char_idxs"][idx]
        self.ques_idxs = data["ques_idxs"][idx]
        self.ques_char_idxs = data["ques_char_idxs"][idx]
        self.y1s = data["y1s"][idx]
        self.y2s = data["y2s"][idx]
        self.ids = data["ids"][idx]
        self.uuids = data["uuids"][idx]

        if version == "v2.0":
            self.impossibles = data["impossibles"][idx].astype(np.int)
        self.num = len(self.ids)

    def __len__(self):
        # return 1000
        return self.num

    def __getitem__(self, idx):
        if self.version == "v2.0":
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


def get_loader(npz_file, batch_size, version):
    if version == "v2.0":

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

    dataset = SQuADDataset(npz_file, batch_size, version)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=5,
        collate_fn=collate,
    )
    return data_loader


def convert_tokens(eval_file, qa_id, pp1, pp2, version):

    answer_dict = {}
    remapped_dict = {}
    if version == "v2.0":
        for qid, p1, p2 in zip(qa_id, pp1, pp2):
            uuid = eval_file[str(qid)]["uuid"]
            if p1 == -1 and p2 == -1:
                answer_dict[str(qid)] = ""
                remapped_dict[uuid] = ""
            else:
                try:
                    context = eval_file[str(qid)]["context"]
                    spans = eval_file[str(qid)]["spans"]
                    start_idx = spans[p1][0]
                    end_idx = spans[p2][1]
                    answer_dict[str(qid)] = context[start_idx:end_idx]
                    remapped_dict[uuid] = context[start_idx:end_idx]
                except Exception as e:
                    import ipdb

                    ipdb.set_trace()
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

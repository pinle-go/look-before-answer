import os
import logging
import json
from codecs import open
from collections import Counter

import numpy as np
import spacy
from tqdm import tqdm

"""
The content of this file is mostly copied from https://github.com/HKUST-KnowComp/R-Net/blob/master/prepro.py
"""
nlp = spacy.blank("en")


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def _get_answer_span(answer, spans, texts):
    text = answer["text"]
    start = answer["answer_start"]
    end = start + len(text)
    texts.append(text)
    answer_span = []
    # this loop finds the overlap of answer and context
    for idx, span in enumerate(spans):
        if not (end <= span[0] or start >= span[1]):
            answer_span.append(idx)
    return answer_span[0], answer_span[-1]


def keep_unique_answers(y1, y2):
    if len(y1) > 0:
        a, b = zip(*list(set([(i, j) for i, j in zip(y1, y2)])))
        return a, b
    return y1, y2


def process_file(filename, data_type, word_counter, char_counter, version="v2.0"):
    """
        filename: json file to read
        data_type : 'train'/'test'/'dev'
        word_counter: Just a counter for word occurence
        char_counter: Just a counter for char
    """

    print("Generating {} examples...\n".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                # tokenize the para and store the span of each token in spans
                # we store spans because we get position of answer start and the answer in the data
                context = para["context"].replace("''", '" ').replace("``", '" ')
                context_tokens = word_tokenize(context)
                spans = convert_idx(context, context_tokens)
                context_chars = [list(token) for token in context_tokens]
                for token in context_tokens:
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace("''", '" ').replace("``", '" ')
                    ques_tokens = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    if version == "v2.0":
                        y1s, y2s = [], []
                        answer_texts = []
                        plausible_y1s, plausible_y2s = [], []
                        plausible_answer_texts = []
                        is_impossible = bool(qa["is_impossible"])

                        # if answering is impossible, some qas might have plausible answer and we record that.
                        if is_impossible:
                            for answer in qa["plausible_answers"]:
                                y1, y2 = _get_answer_span(
                                    answer, spans, plausible_answer_texts
                                )
                                plausible_y1s.append(y1)
                                plausible_y2s.append(y2)
                            plausible_y1s, plausible_y2s = keep_unique_answers(
                                plausible_y1s, plausible_y2s
                            )

                        else:
                            for answer in qa["answers"]:
                                y1, y2 = _get_answer_span(answer, spans, answer_texts)
                                y1s.append(y1)
                                y2s.append(y2)
                            y1s, y2s = keep_unique_answers(y1s, y2s)
                        example = {
                            "context_tokens": context_tokens,
                            "context_chars": context_chars,
                            "ques_tokens": ques_tokens,
                            "ques_chars": ques_chars,
                            "y1s": y1s,
                            "y2s": y2s,
                            "plausible_y1s": plausible_y1s,
                            "plausible_y2s": plausible_y2s,
                            "id": total,
                            "uuid": qa["id"],
                            "is_impossible": is_impossible,
                        }
                        examples.append(example)

                        eval_examples[str(total)] = {
                            "context": context,
                            "spans": spans,
                            "answers": answer_texts,
                            "plausible_answers": plausible_answer_texts,
                            "uuid": qa["id"],
                            "is_impossible": is_impossible,
                        }
                    elif version == "v1.1":  # v1.1 case
                        y1s, y2s = [], []
                        answer_texts = []

                        for answer in qa["answers"]:
                            y1, y2 = _get_answer_span(answer, spans, answer_texts)
                            y1s.append(y1)
                            y2s.append(y2)
                        y1s, y2s = keep_unique_answers(y1s, y2s)
                        example = {
                            "context_tokens": context_tokens,
                            "context_chars": context_chars,
                            "ques_tokens": ques_tokens,
                            "ques_chars": ques_chars,
                            "y1s": y1s,
                            "y2s": y2s,
                            "id": total,
                            "uuid": qa["id"],
                        }
                        examples.append(example)
                        # note eval files are now indexed by uuid here
                        eval_examples[str(total)] = {
                            "context": context,
                            "spans": spans,
                            "answers": answer_texts,
                            "uuid": qa["id"],
                        }
        print(f"{len(examples)} questions in total")
    return examples, eval_examples


def get_embedding(counter, data_type, limit=-1, emb_file=None, vec_size=None):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    # load from file if there is
    if emb_file is not None:
        assert vec_size is not None
        with open(emb_file, "r") as fh:
            for line in tqdm(fh):
                array = line.split()
                l = len(array)
                word = "".join(array[0 : l - vec_size])
                vector = list(map(float, array[l - vec_size : l]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print(
            "{} / {} tokens have corresponding {} embedding vector".format(
                len(embedding_dict), len(filtered_elements), data_type
            )
        )
    # random embedding initialization
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [
                np.random.normal(scale=0.1) for _ in range(vec_size)
            ]
        print(
            "{} tokens have corresponding embedding vector".format(
                len(filtered_elements)
            )
        )

    # NULL and OOV are index 0 and 1 and zero vectors
    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0.0 for _ in range(vec_size)]
    embedding_dict[OOV] = [0.0 for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token] for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def convert_to_features(config, data, word2idx_dict, char2idx_dict):
    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return 1

    def filter_func(example):
        return (
            len(example["context_tokens"]) > para_limit
            or len(example["ques_tokens"]) > ques_limit
        )

    example = {}
    context, question = data
    context = context.replace("''", '" ').replace("``", '" ')
    question = question.replace("''", '" ').replace("``", '" ')
    example["context_tokens"] = word_tokenize(context)
    example["ques_tokens"] = word_tokenize(question)
    example["context_chars"] = [list(token) for token in example["context_tokens"]]
    example["ques_chars"] = [list(token) for token in example["ques_tokens"]]
    spans = convert_idx(context, example["context_tokens"])

    para_limit = config.para_limit
    ques_limit = config.ques_limit
    ans_limit = config.ans_limit
    char_limit = config.char_limit

    if filter_func(example):
        print(" Warning: Context/Question length is over the limit")

    context_idxs = np.zeros([para_limit], dtype=np.int32)
    context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
    ques_idxs = np.zeros([ques_limit], dtype=np.int32)
    ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
    y1 = np.zeros([para_limit], dtype=np.float32)
    y2 = np.zeros([para_limit], dtype=np.float32)

    for i, token in enumerate(example["context_tokens"][:para_limit]):
        context_idxs[i] = _get_word(token)

    for i, token in enumerate(example["ques_tokens"][:ques_limit]):
        ques_idxs[i] = _get_word(token)

    for i, token in enumerate(example["context_chars"][:para_limit]):
        for j, char in enumerate(token[:char_limit]):
            context_char_idxs[i, j] = _get_char(char)

    for i, token in enumerate(example["ques_chars"][:ques_limit]):
        for j, char in enumerate(token[:char_limit]):
            ques_char_idxs[i, j] = _get_char(char)

    return context_idxs, context_char_idxs, ques_idxs, ques_char_idxs, spans


def build_features(
    config, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False
):
    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return word2idx_dict["--OOV--"]

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return char2idx_dict["--OOV--"]

    def filter_func(example, is_test=False):
        # in case of test filter nothing
        if is_test:
            return False
        if version == "v2.0":
            if example["is_impossible"]:
                return (
                    len(example["context_tokens"]) > para_limit
                    or len(example["ques_tokens"]) > ques_limit
                )

        return (
            len(example["context_tokens"]) > para_limit
            or len(example["ques_tokens"]) > ques_limit
            or (example["y2s"][-1] - example["y1s"][-1]) > ans_limit
        )

    para_limit = config.para_limit
    ques_limit = config.ques_limit
    ans_limit = config.ans_limit
    char_limit = config.char_limit
    version = config.version

    print(f"Processing {data_type} examples...")
    total = 0
    meta = {}
    N = len(examples)
    context_idxs = []
    context_char_idxs = []
    ques_idxs = []
    ques_char_idxs = []
    y1s = []
    y2s = []
    ids = []
    uuids = []
    id_to_uuid = {}
    if version == "v2.0":
        impossibles = []
    for n, example in tqdm(enumerate(examples)):
        # if filter returns true, then move to next example
        if filter_func(example, is_test):
            continue
        total += 1
        context_idx = np.zeros([para_limit], dtype=np.int32)
        context_char_idx = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idx = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idx = np.zeros([ques_limit, char_limit], dtype=np.int32)

        for i, token in enumerate(example["context_tokens"][:para_limit]):
            context_idx[i] = _get_word(token)
        for i, token in enumerate(example["ques_tokens"][:ques_limit]):
            ques_idx[i] = _get_word(token)

        for i, token in enumerate(example["context_chars"][:para_limit]):
            for j, char in enumerate(token[:char_limit]):
                context_char_idx[i, j] = _get_char(char)
        for i, token in enumerate(example["ques_chars"][:ques_limit]):
            for j, char in enumerate(token[:char_limit]):
                ques_char_idx[i, j] = _get_char(char)

        if version == "v2.0":
            if not example["is_impossible"]:
                starts, ends = example["y1s"], example["y2s"]
            elif config.use_plausible is True and len(example["plausible_y1s"]) > 0:
                starts, ends = example["plausible_y1s"], example["plausible_y2s"]
            else:
                starts, ends = [-1], [-1]

            # append one example for each possible answer
            for start, end in zip(starts, ends):
                ques_char_idxs.append(ques_char_idx)
                context_idxs.append(context_idx)
                ques_idxs.append(ques_idx)
                context_char_idxs.append(context_char_idx)
                y1s.append(start)
                y2s.append(end)
                ids.append(example["id"])
                impossibles.append(example["is_impossible"])
                uuids.append(example["uuid"])
                id_to_uuid[example["id"]] = example["uuid"]
        else:
            starts, ends = example["y1s"], example["y2s"]

            for start, end in zip(starts, ends):
                ques_char_idxs.append(ques_char_idx)
                context_idxs.append(context_idx)
                ques_idxs.append(ques_idx)
                context_char_idxs.append(context_char_idx)
                y1s.append(start)
                y2s.append(end)
                ids.append(example["id"])
                uuids.append(example["uuid"])
                id_to_uuid[example["id"]] = example["uuid"]

    if version == "v2.0":
        np.savez(
            out_file,
            context_idxs=np.array(context_idxs),
            context_char_idxs=np.array(context_char_idxs),
            ques_idxs=np.array(ques_idxs),
            ques_char_idxs=np.array(ques_char_idxs),
            y1s=np.array(y1s),
            y2s=np.array(y2s),
            ids=np.array(ids),
            impossibles=np.array(impossibles),
            uuids=np.array(uuids),
        )
    else:
        np.savez(
            out_file,
            context_idxs=np.array(context_idxs),
            context_char_idxs=np.array(context_char_idxs),
            ques_idxs=np.array(ques_idxs),
            ques_char_idxs=np.array(ques_char_idxs),
            y1s=np.array(y1s),
            y2s=np.array(y2s),
            ids=np.array(ids),
            uuids=np.array(uuids),
        )
    print("Built {} / {} instances of features in total".format(len(y1s), N))
    print("Processed {} instances of features in total".format(total))

    meta["total"] = len(y1s)
    meta["id_to_uuid"] = id_to_uuid
    return meta


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh, indent=4, sort_keys=True)


def preprocess(args, config):
    word_counter, char_counter = Counter(), Counter()

    # get embeddings
    word_emb_file = config.glove_word_file
    char_emb_file = config.glove_char_file if config.pretrained_char else None

    # handle train file
    train_examples, train_eval = process_file(
        config.raw_train_file, "train", word_counter, char_counter, config.version
    )
    dev_examples, dev_eval = process_file(
        config.raw_dev_file, "dev", word_counter, char_counter, config.version
    )
    if os.path.exists(config.raw_test_file):
        test_examples, test_eval = process_file(
            config.raw_test_file, "test", word_counter, char_counter
        )

    # Note that we are getting embeddings for as much as data as possible (train/test/dev) while training.
    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, "word", emb_file=word_emb_file, vec_size=config.word_emb_dim
    )
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, "char", emb_file=char_emb_file, vec_size=config.char_emb_dim
    )

    build_features(
        config, train_examples, "train", config.train_file, word2idx_dict, char2idx_dict
    )
    dev_meta = build_features(
        config,
        dev_examples,
        "dev",
        config.dev_file,
        word2idx_dict,
        char2idx_dict,
        is_test=True,
    )
    if os.path.exists(config.raw_test_file):
        test_meta = build_features(
            config,
            test_examples,
            "test",
            config.test_record_file,
            word2idx_dict,
            char2idx_dict,
            is_test=True,
        )

    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.char_emb_file, char_emb_mat, message="char embedding")
    save(config.word2idx_file, word2idx_dict, message="word dictionary")
    save(config.char2idx_file, char2idx_dict, message="char dictionary")
    save(config.train_eval_file, train_eval, message="train eval")
    save(config.dev_eval_file, dev_eval, message="dev eval")
    save(config.dev_meta_file, dev_meta, message="dev meta")

    if os.path.exists(config.raw_test_file):
        save(config.test_eval_file, test_eval, message="test eval")
        save(config.test_meta_file, test_meta, message="test meta")

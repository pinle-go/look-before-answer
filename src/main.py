import argparse
import importlib
import logging
import sys
import importlib.util
import os
import utils.utils as utils
import utils.data as data
import utils.model as model_utils
import utils.evaluation as evaluation
import dill
import tqdm
import json
import numpy as np
import torch
import math
import shutil


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


class Trainer:
    def __init__(self, model, train_dataset, dev_dataset, dev_eval, config):
        self.train_data = train_dataset
        self.dev_data = dev_dataset
        self.dev_eval = dev_eval
        self.config = config
        self.model = model
        self.num_epoch = 0

        def lr_calc(ee):
            cr = config.learning_rate / math.log2(config.lr_warm_up_num)
            if ee < config.lr_warm_up_num:
                return cr * math.log2(ee + 1)
            else:
                return config.learning_rate

        self.optimizer = torch.optim.Adam(
            lr=1.0,
            betas=(0.9, 0.999),
            eps=1e-7,
            weight_decay=config.weight_decay,
            params=filter(lambda param: param.requires_grad, model.parameters()),
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda x: lr_calc(x)
        )
        self.loss_func = model_utils.get_loss_func(config.model_type, config.version)

        self.ema = EMA(config.decay)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ema.register(name, param.data)

    def train_epoch(self, epoch_num, best_loss, no_improvement_step):
        self.model.train()
        device, version, checkpoint = (
            self.config.device,
            self.config.version,
            self.config.checkpoint,
        )
        losses = []
        for i, batch in enumerate(self.train_data):
            loss = self.train_iter(
                batch, i + epoch_num * len(self.train_data), version, device
            )
            loss = loss.cpu().detach().numpy()
            print(f"\riteration: {i}/{len(self.train_data)}; loss:{loss}", end="")
            losses.append(loss)

            if (i + 1) % checkpoint == 0 and (i + 1) < checkpoint * (
                len(self.train_data) // checkpoint
            ):
                self.ema.assign(self.model)
                self.model.eval()
                metrics = evaluate(
                    self.model, self.dev_data, self.dev_eval, self.config
                )
                self.ema.resume(self.model)
                self.model.train()

                if metrics["loss"] < best_loss:
                    best_loss = metrics["loss"]
                    no_improvement_step = 0
                else:
                    no_improvement_step += 1
                print(
                    f"Best loss: {best_loss}, No improvement since: {no_improvement_step}"
                )

                if no_improvement_step >= self.config.patience:
                    return np.mean(losses), True, best_loss, no_improvement_step

        self.ema.assign(self.model)
        self.model.eval()
        metrics = evaluate(self.model, self.dev_data, self.dev_eval, self.config)
        self.ema.resume(self.model)
        self.model.train()

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            no_improvement_step = 0
        else:
            no_improvement_step += 1
        print(f"Best loss: {best_loss}, No improvement since: {no_improvement_step}")

        if no_improvement_step >= self.config.patience:
            return np.mean(losses), True

        return np.mean(losses), False, best_loss, no_improvement_step

    def train_iter(self, batch, iter, version, device):
        if version == "v2.0":
            Cwid, Ccid, Qwid, Qcid, y1, y2, ids, impossibles = batch
        else:
            Cwid, Ccid, Qwid, Qcid, y1, y2, ids = batch
        Cwid, Ccid, Qwid, Qcid = (
            Cwid.to(device),
            Ccid.to(device),
            Qwid.to(device),
            Qcid.to(device),
        )
        y1, y2 = y1.to(device), y2.to(device)

        if version == "v2.0":
            p1, p2, z = self.model(Cwid, Ccid, Qwid, Qcid)
            impossibles = impossibles.to(device)
        else:
            p1, p2 = self.model(Cwid, Ccid, Qwid, Qcid)
            z = impossibles = None

        loss = self.loss_func(p1, p2, y1, y2, z, impossibles, device=self.config.device)
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        self.ema(self.model, iter)
        return loss

    def train(self):
        best_loss = 10000
        no_improvement_step = 0
        for i in range(self.num_epoch, self.config.max_epochs):
            self.num_epoch = i
            print(f"Training epoch {i}")
            loss, stop, best_loss, no_improvement_step = self.train_epoch(
                i, best_loss, no_improvement_step
            )
            print(f"epoch: {i}; loss : {loss}")
            if stop:
                print(
                    f"No model improvement even after {self.config.patience} steps. Stopping!!"
                )
                break
            if (i + 1) % self.config.save_every and i:
                self.save(self.config.model_fname)
        self.save(self.config.model_fname)
        metrics = evaluate(self.model, self.dev_data, self.dev_eval, self.config)

        print(f"Final results F1: {metrics.get('f1')}")
        print(f"Exact Match: {metrics.get('exact_match')}")
        print(f"Answerability {metrics.get('answerability_acc')}")

    def save(self, fname):
        data = {}
        # we always save the EMAed model
        self.ema.assign(self.model)
        data["model"] = self.model.state_dict()
        data["optimizer"] = self.optimizer.state_dict()
        data["scheduler"] = self.scheduler.state_dict()
        data["num_epoch"] = self.num_epoch
        data["ema"] = self.ema
        torch.save(data, fname, pickle_module=dill)
        self.ema.resume(self.model)

    def load(self, fname):
        data = torch.load(fname, pickle_module=dill)
        state_dict = self.model.state_dict()
        state_dict.update(data["model"])
        self.model.load_state_dict(state_dict)

        state_dict = self.optimizer.state_dict()
        state_dict.update(data["optimizer"])
        self.optimizer.load_state_dict(state_dict)

        state_dict = self.scheduler.state_dict()
        state_dict.update(data["scheduler"])
        self.scheduler.load_state_dict(state_dict)

        self.ema = data["ema"]
        self.num_epoch = data["num_epoch"]


def train(args, config):
    # load data
    print("Loading data")
    train_dataset = data.get_loader(
        config.train_file, config.batch_size, config.version
    )
    dev_dataset = data.get_loader(
        config.dev_file, config.val_batch_size, config.version
    )
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)

    print("loading embeddings")
    with open(config.word_emb_file, "rb") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "rb") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)

    # create model
    print("creating model")
    Model = model_utils.get_model_func(config.model_type, config.version)
    model = Model(word_mat, char_mat, config).to(config.device)
    model = torch.nn.DataParallel(model)

    print("Training Model")
    trainer = Trainer(model, train_dataset, dev_dataset, dev_eval_file, config)
    if args.model_file is not None:
        trainer.load(args.model_file)
        trainer.ema.resume(trainer.model)
    trainer.train()


def evaluate(model, dataset, eval_file, config):
    print()
    model.eval()
    answer_dict_id, answer_dict_uuid = {}, {}
    losses = []
    version, device = config.version, config.device

    loss_func = model_utils.get_loss_func(config.model_type, config.version)
    pred_func = model_utils.get_pred_func(config.model_type, config.version)

    with torch.no_grad():
        for i, batch in enumerate(dataset):
            if version == "v2.0":
                Cwid, Ccid, Qwid, Qcid, y1, y2, ids, impossibles = batch
            else:
                Cwid, Ccid, Qwid, Qcid, y1, y2, ids = batch

            Cwid, Ccid, Qwid, Qcid = (
                Cwid.to(device),
                Ccid.to(device),
                Qwid.to(device),
                Qcid.to(device),
            )
            y1, y2 = y1.to(device), y2.to(device)

            # compute loss and impossible
            if version == "v2.0":
                p1, p2, z = model(Cwid, Ccid, Qwid, Qcid)
                impossibles = impossibles.to(device)
            else:
                p1, p2 = model(Cwid, Ccid, Qwid, Qcid)
                z = impossibles = None

            loss = loss_func(p1, p2, y1, y2, z, impossibles, device=config.device)
            ymin, ymax = pred_func(p1, p2, z)

            losses.append(loss)
            answer_dict_id_, answer_dict_uuid_ = data.convert_tokens(
                eval_file,
                ids.tolist(),
                ymin.tolist(),
                ymax.tolist(),
                version=config.version,
            )

            answer_dict_id.update(answer_dict_id_)
            answer_dict_uuid.update(answer_dict_uuid_)

            print(
                "\rSTEP {:8d}/{} loss {:8f}".format(i + 1, len(dataset), loss.item()),
                end="",
            )

    loss = np.mean(losses)
    # eval file is indexed by id so we use id dictionary
    metrics = evaluation.evaluate(eval_file, answer_dict_id, config.version)
    with open(f"{config.answer_file}", "w") as f:
        json.dump(answer_dict_uuid, f, sort_keys=True, indent=4)

    metrics["loss"] = loss
    print()
    if version == "v2.0":
        print(
            "EVAL loss {:8f} F1 {:8f} EM {:8f} answer possible {:8f}\n".format(
                loss,
                metrics["f1"],
                metrics["exact_match"],
                metrics["answerability_acc"],
            )
        )
    else:
        print(
            "EVAL loss {:8f} F1 {:8f} EM {:8f}\n".format(
                loss, metrics["f1"], metrics["exact_match"]
            )
        )
    return metrics


def test(args, config):

    from preprocess import convert_to_features

    version, device = config.version, config.device
    print("loading embeddings")
    with open(config.word_emb_file, "rb") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "rb") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)

    # create model
    print("creating model")
    Model = model_utils.get_model_func(config.model_type, version)
    pred_func = model_utils.get_pred_func(config.model_type, version)

    model = Model(word_mat, char_mat, config).to(device)
    model = torch.nn.DataParallel(model)

    print("Loading Model")
    data = torch.load(args.model_file, map_location="cpu", pickle_module=dill)
    state_dict = model.state_dict()
    state_dict.update(data["model"])
    model.load_state_dict(state_dict)

    with open(args.test_file, "r") as f:
        test_data = json.load(f)
    with open(config.word2idx_file) as f:
        word2idx_dict = json.load(f)
    with open(config.char2idx_file) as f:
        char2idx_dict = json.load(f)

    print("Extracting examples")
    examples = []
    for article in test_data["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                ques = qa["question"]
                c_idx, c_char_idx, q_idx, q_char_idx, span = convert_to_features(
                    config, (context, ques), word2idx_dict, char2idx_dict
                )
                uuid = qa["id"]
                example = {
                    "context_tokens": c_idx,
                    "context_chars": c_char_idx,
                    "ques_tokens": q_idx,
                    "ques_chars": q_char_idx,
                    "uuid": uuid,
                    "span": span,
                    "context": context,
                }
                examples.append(example)
    N = len(examples)
    answer_dict = {}
    with torch.no_grad():
        for i in range(0, N, config.val_batch_size):
            print(f"\r current index {i}/{N}", end="")
            Cwid = np.concatenate(
                list(
                    map(
                        lambda x: np.expand_dims(x["context_tokens"], axis=0),
                        examples[i : i + config.val_batch_size],
                    )
                ),
                axis=0,
            )
            Ccid = np.concatenate(
                list(
                    map(
                        lambda x: np.expand_dims(x["context_chars"], axis=0),
                        examples[i : i + config.val_batch_size],
                    )
                ),
                axis=0,
            )
            Qwid = np.concatenate(
                list(
                    map(
                        lambda x: np.expand_dims(x["ques_tokens"], axis=0),
                        examples[i : i + config.val_batch_size],
                    )
                ),
                axis=0,
            )
            Qcid = np.concatenate(
                list(
                    map(
                        lambda x: np.expand_dims(x["ques_chars"], axis=0),
                        examples[i : i + config.val_batch_size],
                    )
                ),
                axis=0,
            )

            Cwid, Ccid, Qwid, Qcid = (
                torch.tensor(Cwid, device=device).long(),
                torch.tensor(Ccid, device=device).long(),
                torch.tensor(Qwid, device=device).long(),
                torch.tensor(Qcid, device=device).long(),
            )

            # compute loss and impossible
            if version == "v2.0":
                p1, p2, z = model(Cwid, Ccid, Qwid, Qcid)
            else:
                p1, p2 = model(Cwid, Ccid, Qwid, Qcid)
                z = None
            ymin, ymax = pred_func(p1, p2, z)

            if version == "v2.0":
                for p1, p2, example in zip(
                    ymin, ymax, examples[i : i + config.val_batch_size]
                ):
                    uuid, span, context = (
                        example["uuid"],
                        example["span"],
                        example["context"],
                    )
                    if p1 == -1 and p2 == -1:
                        answer = ""
                    else:
                        answer = context[span[p1][0] : span[p2][1]]
                    answer_dict[uuid] = answer
            else:
                for p1, p2, example in zip(
                    ymin, ymax, examples[i : i + config.val_batch_size]
                ):
                    uuid, span, context = (
                        example["uuid"],
                        example["span"],
                        example["context"],
                    )
                    answer = context[span[p1][0] : span[p2][1]]
                    answer_dict[uuid] = answer
    with open(f"{config.answer_file}", "w") as f:
        json.dump(answer_dict, f, sort_keys=True, indent=4)


def main(args, config):
    if args.mode == "preprocess":
        from preprocess import preprocess

        try:
            utils.makedirs("data/", raise_error=True)
        except Exception:
            print(
                f"Warning : data folder already exists. Some data may get overwritten"
            )
        print("Running preprocessing")
        preprocess(args, config)

    elif args.mode == "train":
        try:
            utils.makedirs(args.output_folder, raise_error=True)
        except Exception:
            print(
                f"Warning : {args.output_folder} already exists. Some data may get overwritten"
            )
        shutil.copytree("src/", f"{args.output_folder}/src")
        shutil.copy(args.config_file, args.output_folder)
        train(args, config)

    elif args.mode == "test":
        if args.model_file is None or args.test_file is None:
            print("Error: Provide model_file and test_file file")
            sys.exit(1)

        try:
            utils.makedirs(args.output_folder, raise_error=True)
        except Exception:
            print(
                f"Warning : {args.output_folder} already exists. Some data may get overwritten"
            )
        test(args, config)


if __name__ == "__main__":
    FORMAT = "%(asctime)-15s : %(levelname)s : %(filename)s : %(lineno)d : %(message)s"

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", required=True)
    parser.add_argument("--mode", choices=["train", "test", "preprocess"])
    parser.add_argument("-o", "--output_folder", required=True)
    parser.add_argument("-m", "--model_file")
    parser.add_argument("-t", "--test_file")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["INFO", "DEBUG", "ERROR", "WARNING", "CRITICAL"],
    )

    args = parser.parse_args()

    # logger setup
    # TODO: fix logger. It doesn't work with TQDM >.<
    # class TqdmLoggingHandler(logging.Handler):
    #     def __init__(self, level=logging.NOTSET):
    #         super(self.__class__, self).__init__(level)

    #     def emit(self, record):
    #         try:
    #             msg = self.format(record)
    #             tqdm.tqdm.write(msg)
    #             self.flush()
    #         except (KeyboardInterrupt, SystemExit):
    #             raise
    #         except:
    #             self.handleError(record)

    # logging.basicConfig(format=FORMAT)
    # logger = logging.getLogger(__name__)
    # logger.setLevel(args.log_level)
    # logger.addHandler(TqdmLoggingHandler())

    # load config
    spec = importlib.util.spec_from_file_location("config", args.config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    config = config.config
    config["answer_file"] = f"{args.output_folder}/answer.json"
    config["model_fname"] = f"{args.output_folder}/{config['model_fname']}"

    config = argparse.Namespace(**config)

    main(args, config)

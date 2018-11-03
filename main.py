import json
import math
import os
import pickle
import random
import re
import string

import numpy as np
import torch
import torch.cuda
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from absl import app
from tensorboardX import SummaryWriter

from config import config, device
from preproc import preproc
from model_utils import get_pred_func, get_loss_func
from utils import EMA, get_loader, convert_tokens

writer = SummaryWriter("/tmp/tnsr")
"""
Some functions are from the official evaluation script.
"""


def train(
    model,
    optimizer,
    scheduler,
    dataset,
    dev_dataset,
    dev_eval_file,
    start,
    ema,
    loss_func,
):
    model.train()
    losses = []
    print(f"Training epoch {start}")
    for i, batch in enumerate(dataset):
        if config.data_version == "V2":
            Cwid, Ccid, Qwid, Qcid, y1, y2, ids, impossibles = batch
        else:
            Cwid, Ccid, Qwid, Qcid, y1, y2, ids = batch

        optimizer.zero_grad()
        Cwid, Ccid, Qwid, Qcid = (
            Cwid.to(device),
            Ccid.to(device),
            Qwid.to(device),
            Qcid.to(device),
        )
        y1, y2 = y1.to(device), y2.to(device)
    
        if config.data_version == "V2":
            p1, p2, z = model(Cwid, Ccid, Qwid, Qcid)
            impossibles = impossibles.to(device)        
            if config.model_type == "model0":
                # in this debug model, we basic ignore all no-answer questions
                p1, p2 = p1[impossibles == 0], p2[impossibles == 0]
                y1, y2 = y1[impossibles == 0], y2[impossibles == 0]

                p1 = F.log_softmax(p1, dim=1)
                p2 = F.log_softmax(p2, dim=1)
                loss1 = F.nll_loss(p1, y1)
                loss2 = F.nll_loss(p2, y2)
                loss = loss1 + loss2
                writer.add_scalar("data/loss", loss.item(), i + start * len(dataset))
            elif config.model_type == "model1":
                pass
            elif config.model_type == "model2":
                pass
            elif config.model_type == "model3":
                pass
            else:
                raise ValueError()
        else:
            p1, p2 = model(Cwid, Ccid, Qwid, Qcid)
            p1 = F.log_softmax(p1, dim=1)
            p2 = F.log_softmax(p2, dim=1)
            loss1 = F.nll_loss(p1, y1)
            loss2 = F.nll_loss(p2, y2)
            loss = loss1 + loss2
            writer.add_scalar("data/loss", loss.item(), i + start * len(dataset))

        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), config.grad_clip)
        optimizer.step()

        ema(model, i + start * len(dataset))

        scheduler.step()
        if (i + 1) % config.checkpoint == 0 and (i + 1) < config.checkpoint * (
            len(dataset) // config.checkpoint
        ):
            ema.assign(model)
            metrics = test(
                model,
                dev_dataset,
                dev_eval_file,
                i + start * len(dataset),
                get_loss_func(),
                get_pred_func(),
            )
            ema.resume(model)
            model.train()
        for param_group in optimizer.param_groups:
            # print("Learning:", param_group['lr'])
            writer.add_scalar("data/lr", param_group["lr"], i + start * len(dataset))
        print(
            "\rSTEP {:8d}/{} loss {:8f}".format(i + 1, len(dataset), loss.item()),
            end="",
        )
    loss_avg = np.mean(losses)
    print("STEP {:8d} Avg_loss {:8f}\n".format(start, loss_avg))


def test(model, dataset, eval_file, test_i, loss_func, pred_func):
    print("\nTest")
    model.eval()
    answer_dict = {}
    losses = []
    num_batches = config.val_num_batches
    with torch.no_grad():
        for i, batch in enumerate(dataset):
            if config.data_version == "V2":
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
            if config.data_version == "V2":
                p1, p2, z = model(Cwid, Ccid, Qwid, Qcid)
                impossibles = impossibles.to(device)
            
                if config.model_type == "model0":
                    ymin, ymax = [], []
                    for p1_, p2_, z_, y1_, y2_, impossibles_ in zip(
                        p1, p2, z, y1, y2, impossibles
                    ):
                        if z_ < 0.5:  # answerable
                            outer = torch.matmul(p1_.unsqueeze(1), p2_.unsqueeze(0))
                            outer = torch.triu(outer)
                            a1, _ = torch.max(outer, dim=1)
                            a2, _ = torch.max(outer, dim=0)
                            ymin_ = torch.argmax(a1, dim=0)
                            ymax_ = torch.argmax(a2, dim=0)
                        else:
                            ymin_ = -1
                            ymax_ = -1
                        ymin.append(ymin_)
                        ymax.append(ymax_)
                    ymin, ymax = torch.LongTensor(ymin), torch.LongTensor(ymax)
                    loss, losses = torch.FloatTensor([0]), [0]
                elif config.model_type == "model1":
                    pass
                elif config.model_type == "model2":
                    pass
                elif config.model_type == "model3":
                    pass
                else:
                    raise ValueError()
            else:
                p1, p2 = model(Cwid, Ccid, Qwid, Qcid)
                p1 = F.log_softmax(p1, dim=1)
                p2 = F.log_softmax(p2, dim=1)
                loss1 = F.nll_loss(p1, y1)
                loss2 = F.nll_loss(p2, y2)
                loss = torch.mean(loss1 + loss2)
                losses.append(loss.item())

                p1 = F.softmax(p1, dim=1)
                p2 = F.softmax(p2, dim=1)

                # ymin = []
                # ymax = []
                outer = torch.matmul(p1.unsqueeze(2), p2.unsqueeze(1))
                for j in range(outer.size()[0]):
                    outer[j] = torch.triu(outer[j])
                    # outer[j] = torch.tril(outer[j], config.ans_limit)
                a1, _ = torch.max(outer, dim=2)
                a2, _ = torch.max(outer, dim=1)
                ymin = torch.argmax(a1, dim=1)
                ymax = torch.argmax(a2, dim=1)

            losses.append(loss)
            answer_dict_, _ = convert_tokens(
                eval_file,
                ids.tolist(),
                ymin.tolist(),
                ymax.tolist(),
                zz=(z.tolist() if config.data_version == "V2" else None),
            )
            answer_dict.update(answer_dict_)
            print(
                "\rSTEP {:8d}/{} loss {:8f}".format(i + 1, len(dataset), loss.item()),
                end="",
            )
            if (i + 1) == num_batches:
                break

    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    f = open("log/answers.json", "w")
    json.dump(answer_dict, f)
    f.close()
    metrics["loss"] = loss
    print(
        "EVAL loss {:8f} F1 {:8f} EM {:8f}\n".format(
            loss, metrics["f1"], metrics["exact_match"]
        )
    )
    if config.mode == "train":
        writer.add_scalar("data/test_loss", loss, test_i)
        writer.add_scalar("data/F1", metrics["f1"], test_i)
        writer.add_scalar("data/EM", metrics["exact_match"], test_i)
    return metrics


def train_entry(config):
    from models import QANet, QANetV0

    if config.model_type == "model0":
        QANet = QANetV0
    if config.model_type == "model1":
        QANet = QANet
        

    with open(config.word_emb_file, "rb") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "rb") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)

    print("Building model...")

    train_dataset = get_loader(config.train_record_file, config.batch_size)
    dev_dataset = get_loader(config.dev_record_file, config.batch_size)

    lr = config.learning_rate
    base_lr = 1
    lr_warm_up_num = config.lr_warm_up_num

    model = QANet(word_mat, char_mat).to(device)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)

    ema = EMA(config.decay)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)

    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = optim.Adam(
        lr=base_lr, betas=(0.9, 0.999), eps=1e-7, weight_decay=5e-8, params=parameters
    )
    cr = lr / math.log2(lr_warm_up_num)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ee: cr * math.log2(ee + 1) if ee < lr_warm_up_num else lr,
    )
    best_f1 = 0
    best_em = 0
    patience = 0
    unused = False
    for iter in range(config.num_epoch):
        train(
            model,
            optimizer,
            scheduler,
            train_dataset,
            dev_dataset,
            dev_eval_file,
            iter,
            ema,
            get_loss_func(),
        )
        ema.assign(model)
        metrics = test(
            model,
            dev_dataset,
            dev_eval_file,
            (iter + 1) * len(train_dataset),
            get_loss_func(),
            get_pred_func(),
        )
        dev_f1 = metrics["f1"]
        dev_em = metrics["exact_match"]
        if dev_f1 < best_f1 and dev_em < best_em:
            patience += 1
            if patience > config.early_stop:
                break
        else:
            patience = 0
            best_f1 = max(best_f1, dev_f1)
            best_em = max(best_em, dev_em)

        fn = os.path.join(config.save_dir, "model.pt")
        torch.save(model, fn)
        ema.resume(model)


def test_entry(config):
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    dev_dataset = get_loader(config.dev_record_file, config.batch_size)
    fn = os.path.join(config.save_dir, "model.pt")
    model = torch.load(fn)
    test(model, dev_dataset, dev_eval_file, 0, get_loss_func(), get_pred_func())


def main(_):
    if config.mode == "train":
        train_entry(config)
    elif config.mode == "data":
        preproc(config)
    elif config.mode == "debug":
        config.batch_size = 2
        # config.num_steps = 32
        config.val_num_batches = 2
        config.checkpoint = 2
        config.period = 1
        train_entry(config)
    elif config.mode == "test":
        test_entry(config)
    else:
        print("Unknown mode")
        exit(0)


if __name__ == "__main__":
    app.run(main)

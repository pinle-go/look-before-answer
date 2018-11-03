import torch
from torch.nn import functional as F

from config import config


def loss_origin(p1, p2, y1, y2, z, impossibles):
    p1 = F.log_softmax(p1, dim=1)
    p2 = F.log_softmax(p2, dim=1)
    loss1 = F.nll_loss(p1, y1)
    loss2 = F.nll_loss(p2, y2)
    loss = loss1 + loss2
    return loss


def loss_model0(p1, p2, y1, y2, z, impossibles):
    # in this debug model, we basic ignore all no-answer questions
    p1, p2 = p1[impossibles == 0], p2[impossibles == 0]
    y1, y2 = y1[impossibles == 0], y2[impossibles == 0]

    p1 = F.log_softmax(p1, dim=1)
    p2 = F.log_softmax(p2, dim=1)
    loss1 = F.nll_loss(p1, y1)
    loss2 = F.nll_loss(p2, y2)
    loss = loss1 + loss2
    return loss


def loss_model1(p1, p2, y1, y2, z, impossibles):
    # compute loss when possible
    if len(p1[impossibles == 0]) > 0:
        p1_ = torch.log(p1[impossibles == 0])
        p2_ = torch.log(p2[impossibles == 0])
        loss1 = F.nll_loss(p1_, y1[impossibles == 0])
        loss2 = F.nll_loss(p2_, y2[impossibles == 0])
        loss = loss1 + loss2
    else:
        loss = 0

    # compute loss when impossible
    if len(p1[impossibles != 0]) > 0:
        # TODO: something might be wrong here
        loss += -torch.log(z).mean()

    return loss


def pred_origin(p1, p2, z):
    p1 = F.softmax(p1, dim=1)
    p2 = F.softmax(p2, dim=1)
    outer = torch.matmul(p1.unsqueeze(2), p2.unsqueeze(1))
    for j in range(outer.size()[0]):
        outer[j] = torch.triu(outer[j])
    a1, _ = torch.max(outer, dim=2)
    a2, _ = torch.max(outer, dim=1)
    ymin = torch.argmax(a1, dim=1)
    ymax = torch.argmax(a2, dim=1)
    return ymin, ymax


def pred_model0(p1, p2, z):
    ymin, ymax = [], []
    for p1_, p2_, z_ in zip(p1, p2, z):
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
    return ymin, ymax

def pred_model1(p1, p2, z):
    ymin, ymax = [], []
    for p1_, p2_, z_ in zip(p1, p2, z):
        outer = torch.matmul(p1_.unsqueeze(1), p2_.unsqueeze(0))
        outer = torch.triu(outer)
        if outer.max() > z_:
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
    return ymin, ymax


def get_loss_func():
    if config.data_version == "V2":
        if config.model_type == "model0":
            return loss_model0
        elif config.model_type == "model1":
            return loss_model1
        elif config.model_type == "model2":
            raise NotImplementedError()
        elif config.model_type == "model3":
            raise NotImplementedError()
        else:
            raise ValueError()
    else:
        return loss_origin


def get_pred_func():
    if config.data_version == "V2":
        if config.model_type == "model0":
            return pred_model0
        elif config.model_type == "model1":
            return pred_model1
        elif config.model_type == "model2":
            raise NotImplementedError()
        elif config.model_type == "model3":
            raise NotImplementedError()
        else:
            raise ValueError()
    else:
        return pred_origin


def get_model_func():
    from models import QANet, QANetV0, QANetV1

    if config.data_version == "V2":
        if config.model_type == "model0":
            return QANetV0
        elif config.model_type == "model1":
            return QANetV1
        elif config.model_type == "model2":
            raise NotImplementedError()
        elif config.model_type == "model3":
            raise NotImplementedError()
        else:
            raise ValueError()
    else:
        return QANet


__all__ = ["get_loss_func", "get_pred_func", "get_model_func"]

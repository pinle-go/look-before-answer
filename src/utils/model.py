import torch
from torch.nn import functional as F

# TODO review loss functions


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
    Lc = p1.size(1) - 1
    y1[y1 == -1] = Lc
    y2[y2 == -1] = Lc
    p1 = F.log_softmax(p1, dim=1)
    p2 = F.log_softmax(p2, dim=1)

    try:
        loss1 = F.nll_loss(p1, y1)
        loss2 = F.nll_loss(p2, y2)
    except:
        from IPython import embed

        embed()
    loss = loss1 + loss2

    return loss


def loss_model2(p1, p2, y1, y2, z, impossibles):
    sa_max, _ = torch.max(p1, dim=1)
    sb_max, _ = torch.max(p2, dim=1)
    sa, sb = p1 - sa_max.unsqueeze(1), p2 - sb_max.unsqueeze(1)

    exp_sa, exp_sb, exp_z = torch.exp(sa), torch.exp(sb), torch.exp(z)
    normalizer = exp_z + (torch.sum(exp_sa, dim=1) * torch.sum(exp_sb, dim=1)).view(
        -1, 1
    )
    exp_sa, exp_sb, exp_z = exp_sa / normalizer, exp_sb / normalizer, exp_z / normalizer

    N = p1.shape[0]
    loss = torch.tensor(0.0).to(device)
    for i in range(N):
        exp_sa_, exp_sb_, exp_z_ = exp_sa[i], exp_sb[i], exp_z[i, 0]
        y1_, y2_ = y1[i], y2[i]

        if impossibles[i] == 0:
            loss += -torch.log(exp_sa_[y1_] * exp_sb_[y2_])
        else:
            loss += -torch.log(exp_z_)

    return loss / N


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
    p1 = F.softmax(p1, dim=1)
    p2 = F.softmax(p2, dim=1)
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

    for p1_, p2_ in zip(p1, p2):
        outer = torch.matmul(p1_.unsqueeze(1), p2_.unsqueeze(0))
        outer = torch.triu(outer)
        z_ = outer[-1, -1]
        outer = outer[:-1, :-1]
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


def pred_model2(p1, p2, z):
    sa_max, _ = torch.max(p1, dim=1)
    sb_max, _ = torch.max(p2, dim=1)
    sa, sb = p1 - sa_max.unsqueeze(1), p2 - sb_max.unsqueeze(1)

    exp_sa, exp_sb, exp_z = torch.exp(sa), torch.exp(sb), torch.exp(z)
    normalizer = exp_z + (torch.sum(exp_sa, dim=1) * torch.sum(exp_sb, dim=1)).view(
        -1, 1
    )
    exp_sa, exp_sb, exp_z = exp_sa / normalizer, exp_sb / normalizer, exp_z / normalizer

    N = p1.shape[0]
    ymin, ymax = [], []

    for i in range(N):
        exp_sa_, exp_sb_, exp_z_ = exp_sa[i], exp_sb[i], exp_z[i, 0]
        outer = torch.matmul(exp_sa_.unsqueeze(1), exp_sb_.unsqueeze(0))
        outer = torch.triu(outer)
        a1, _ = torch.max(outer, dim=1)
        a2, _ = torch.max(outer, dim=0)
        ymin_ = torch.argmax(a1, dim=0)
        ymax_ = torch.argmax(a2, dim=0)
        prob_answer = outer[ymin_, ymax_]
        prob_no_answer = exp_z_
        if prob_no_answer > prob_answer:  # not answerable
            ymin_ = -1
            ymax_ = -1
        ymin.append(ymin_)
        ymax.append(ymax_)
    ymin, ymax = torch.LongTensor(ymin), torch.LongTensor(ymax)
    return ymin, ymax


def get_loss_func(model_type, version):
    if version == "v2.0":
        if model_type == "model0":
            return loss_model0
        elif model_type == "model1":
            return loss_model1
        elif model_type == "model2":
            return loss_model2
        elif model_type == "model3":
            raise NotImplementedError()
        else:
            raise ValueError()
    else:
        return loss_origin


def get_pred_func(model_type, version):
    if version == "v2.0":
        if model_type == "model0":
            return pred_model0
        elif model_type == "model1":
            return pred_model1
        elif model_type == "model2":
            return pred_model2
        elif model_type == "model3":
            raise NotImplementedError()
        else:
            raise ValueError()
    else:
        return pred_origin


def get_model_func(model_type, version):
    from models import QANet, QANetV0, QANetV1, QANetV2

    if version == "v2.0":
        if model_type == "model0":
            return QANetV0
        elif model_type == "model1":
            return QANetV1
        elif model_type == "model2":
            return QANetV2
        elif model_type == "model3":
            raise NotImplementedError()
        else:
            raise ValueError()
    else:
        return QANet


__all__ = ["get_loss_func", "get_pred_func", "get_model_func"]

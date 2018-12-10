import numpy as np
import torch
from torch.nn import functional as F

# TODO review loss functions


def loss_origin(p1, p2, y1, y2, z, impossibles, **kwargs):
    p1 = F.log_softmax(p1, dim=1)
    p2 = F.log_softmax(p2, dim=1)
    # TODO: should have used para limit
    y1[y1 >= 400] = 400
    y2[y2 >= 400] = 400

    loss1 = F.nll_loss(p1, y1, ignore_index=400)
    loss2 = F.nll_loss(p2, y2, ignore_index=400)
    loss = loss1 + loss2
    return loss


def loss_model0(p1, p2, y1, y2, z, impossibles, **kwargs):
    # in this debug model, we basic ignore all no-answer questions
    p1, p2 = p1[impossibles == 0], p2[impossibles == 0]
    y1, y2 = y1[impossibles == 0], y2[impossibles == 0]

    p1 = F.log_softmax(p1, dim=1)
    p2 = F.log_softmax(p2, dim=1)

    # TODO: should have used para limit
    y1[y1 >= 400] = 400
    y2[y2 >= 400] = 400

    loss1 = F.nll_loss(p1, y1, ignore_index=400)
    loss2 = F.nll_loss(p2, y2, ignore_index=400)
    loss = loss1 + loss2
    return loss


def loss_model1(p1, p2, y1, y2, z, impossibles, **kwargs):
    y1[y1 >= 400] = 401
    y2[y2 >= 400] = 401

    # 400th index is for no-answer
    y1[impossibles == 1] = 400
    y2[impossibles == 1] = 400

    p1 = F.log_softmax(p1, dim=1)
    p2 = F.log_softmax(p2, dim=1)

    loss1 = F.nll_loss(p1, y1, ignore_index=401)
    loss2 = F.nll_loss(p2, y2, ignore_index=401)
    loss = loss1 + loss2

    return loss


def loss_model2(p1, p2, y1, y2, z, impossibles, **kwargs):
    sa_max, _ = torch.max(p1, dim=1)
    sb_max, _ = torch.max(p2, dim=1)
    sa_max = sa_max.view(-1, 1)
    sb_max = sb_max.view(-1, 1)

    max_, _ = torch.max(torch.cat([sa_max, sb_max, z], dim=1), dim=1)
    sa, sb, z = p1 - max_.unsqueeze(1), p2 - max_.unsqueeze(1), z - max_.unsqueeze(1)

    exp_sa, exp_sb, exp_z = torch.exp(sa), torch.exp(sb), torch.exp(z)
    normalizer = exp_z + (torch.sum(exp_sa, dim=1) * torch.sum(exp_sb, dim=1)).view(
        -1, 1
    )
    # exp_sa, exp_sb, exp_z = exp_sa / normalizer, exp_sb / normalizer, exp_z / normalizer

    N = p1.shape[0]
    loss = torch.tensor(0.0).to(p1.device)
    for i in range(N):
        sa_, sb_, z_, norm_ = (sa[i], sb[i], z[i, 0], normalizer[i, 0])
        y1_, y2_ = y1[i], y2[i]

        if y1_ >= 400 or y2_ >= 400:
            # ignore loss calculation
            continue
        elif impossibles[i] == 0:
            loss += -sa_[y1_] - sb_[y2_] + torch.log(norm_)
        else:
            loss += -z_ + torch.log(norm_)

    return loss / N


def loss_model3(p1, p2, y1, y2, z, impossible, **kwargs):
    y1[y1 >= 400] = 400
    y2[y2 >= 400] = 400

    y1[y1 == -1] = 400
    y2[y2 == -1] = 400

    a_coeff, s_coeff = kwargs["a_coeff"], kwargs["s_coeff"]

    p1 = F.log_softmax(p1, dim=1)
    p2 = F.log_softmax(p2, dim=1)

    loss1 = F.nll_loss(p1, y1, ignore_index=400)
    loss2 = F.nll_loss(p2, y2, ignore_index=400)
    span_prediction_loss = loss1 + loss2

    answer_prediction_loss = F.binary_cross_entropy_with_logits(
        z, impossible.view(z.shape).float()
    )
    loss = span_prediction_loss * s_coeff + answer_prediction_loss * a_coeff
    return loss


def loss_model_ao(p1, p2, y1, y2, z, impossible, **kwargs):
    return F.binary_cross_entropy_with_logits(z, impossible.float())


def loss_model5(p1, p2, y1, y2, z, impossibles, **kwargs):
    p1 = F.log_softmax(p1, dim=1)
    p2 = F.log_softmax(p2, dim=1)
    # TODO: should have used para limit
    y1[y1 >= 400] = 400
    y2[y2 >= 400] = 400
    y1[y1 <= -1] = 400
    y2[y2 <= -1] = 400

    loss1 = F.nll_loss(p1, y1, ignore_index=400)
    loss2 = F.nll_loss(p2, y2, ignore_index=400)
    loss = loss1 + loss2
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
    p1 = F.softmax(p1, dim=1)
    p2 = F.softmax(p2, dim=1)

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
    sa_max = sa_max.view(-1, 1)
    sb_max = sb_max.view(-1, 1)

    max_, _ = torch.max(torch.cat([sa_max, sb_max, z], dim=1), dim=1)

    sa, sb, z = p1 - max_.unsqueeze(1), p2 - max_.unsqueeze(1), z - max_.unsqueeze(1)

    exp_sa, exp_sb, exp_z = torch.exp(sa), torch.exp(sb), torch.exp(z)
    normalizer = exp_z + (torch.sum(exp_sa, dim=1) * torch.sum(exp_sb, dim=1)).view(
        -1, 1
    )
    # exp_sa, exp_sb, exp_z = exp_sa / normalizer, exp_sb / normalizer, exp_z / normalizer

    N = p1.shape[0]
    ymin, ymax = [], []

    for i in range(N):
        exp_sa_, exp_sb_, exp_z_, norm_ = (
            exp_sa[i],
            exp_sb[i],
            exp_z[i, 0],
            normalizer[i, 0],
        )
        # Note : we are not dividing by norm, divide by norm_ to get the normalized prob
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


def pred_model3(p1, p2, z):
    ymin, ymax = [], []
    p1 = F.softmax(p1, dim=1)
    p2 = F.softmax(p2, dim=1)
    z = torch.sigmoid(z)

    for p1_, p2_, z_ in zip(p1, p2, z):
        outer = torch.matmul(p1_.unsqueeze(1), p2_.unsqueeze(0))
        outer = torch.triu(outer)
        # since we have binary classification loss on z we can compare to a fixed value
        if z_ < 0.5:
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


def pred_model_ao(p1, p2, z, **kwargs):
    ymin, ymax = [], []
    z = z.tolist()

    ymin_, ymax_ = pred_origin(p1, p2, z)

    for idx, z_ in enumerate(z):
        if z_ < 0:
            # append something random, we only need answerability result from
            ymin.append(0)
            ymax.append(1)
        else:
            ymin.append(-1)
            ymax.append(-1)
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
            return loss_model3
        elif model_type == "model_ao":
            return loss_model_ao
        elif model_type == "model4":
            return loss_model_ao
        elif model_type == "model5":
            return loss_model5
        elif model_type == "model6":
            return loss_model_ao
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
            return pred_model3
        elif model_type == "model_ao":
            return pred_model_ao
        elif model_type == "model4":
            return pred_model_ao
        elif model_type == "model5":
            return pred_origin
        elif model_type == "model6":
            return pred_model_ao
        else:
            raise ValueError()
    else:
        return pred_origin


def get_model_func(model_type, version):
    from models import (
        QANet,
        QANetV0,
        QANetV1,
        QANetV2,
        QANetV3,
        QANetAO,
        QANetAO_learned,
    )

    if version == "v2.0":
        if model_type == "model0":
            return QANetV0
        elif model_type == "model1":
            return QANetV1
        elif model_type == "model2":
            return QANetV2
        elif model_type == "model3":
            return QANetV2
        elif model_type == "model_ao":
            return QANetAO
        elif model_type == "model4":
            return QANetV3
        elif model_type == "model5":
            # this is training model for only predicting answer using 2.0 data
            return QANetV0
        elif model_type == "model6":
            return QANetAO_learned
        else:
            raise ValueError()
    else:
        return QANet


__all__ = ["get_loss_func", "get_pred_func", "get_model_func"]

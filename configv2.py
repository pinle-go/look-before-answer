# note that keys can be over-ridden by args
import os
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

home = os.path.expanduser("~")
data_path = "data/"
version = "v2.0"  # or "v2.0"
raw_data_path = f"{home}/data/squad"

glove_word_file = f"{home}/data/glove/glove.840B.300d.txt"
glove_char_file = f"{home}/data/glove/glove.840B.300d-char.txt"
use_plausible = True


config = {
    "device": device,
    # To use plausible data or not
    # set to false for all except model 5 and model 6
    "use_plausible": use_plausible,
    "pretrained_model": "out_plausible_answering/model.pt",
    #
    "version": version,
    # change the model type to run different models
    # model0 - random
    # model1 - QANET_NAF
    # model2 - QANET-NA
    # model3 - QANET-combined
    # model4 - separate-V4 (model with three different learned attention on decoder)
    # model5  (to generate pre-trained model) - Answering only model trained with plausible answer
    # model6 - Answerability only model trained with plausible answers
    "model_type": "model6",
    #
    "raw_train_file": f"{raw_data_path}/train-{version}.json",
    "raw_dev_file": f"{raw_data_path}/dev-{version}.json",
    "raw_test_file": f"{raw_data_path}/test-{version}.json",
    #
    "train_file": f"{data_path}/train-{version}.npz",
    "dev_file": f"{data_path}/dev-{version}.npz",
    "test_file": f"{data_path}/test-{version}.npz ",
    #
    "word_emb_file": f"{data_path}/word_emb-{version}.pkl",
    "char_emb_file": f"{data_path}/char_emb-{version}.pkl",
    "train_eval_file": f"{data_path}/train_eval-{version}.json",
    "dev_eval_file": f"{data_path}/dev_eval-{version}.json",
    "test_eval_file": f"{data_path}/test_eval-{version}.json",
    "dev_meta_file": f"{data_path}/dev_meta-{version}.json",
    "test_meta_file": f"{data_path}/test_meta-{version}.json",
    "word2idx_file": f"{data_path}/word2idx-{version}.json",
    "char2idx_file": f"{data_path}/char2idx-{version}.json",
    #
    "glove_word_file": glove_word_file,
    "glove_char_file": glove_char_file,
    "pretrained_char": False,
    "char_emb_dim": 200,
    "char_emb_dim_projection": 200,  # when using pre-trained embedding we want char embedding to be projected to this dimension
    "word_emb_dim": 300,
    #
    "para_limit": 400,
    "ques_limit": 50,
    "ans_limit": 30,
    "char_limit": 16,
    #
    "batch_size": 32,
    "val_batch_size": 250,
    "learning_rate": 1e-3,
    "max_epochs": 10,
    "weight_decay": 3 * 1e-7,
    "lr_warm_up_num": 1e3,
    "grad_clip": 10,
    "decay": 0.9999,
    #
    "dropout_char": 0.1,
    "dropout_word": 0.05,
    "dropout": 0.1,
    "enc_filters": 128,
    "attention_heads": 1,
    # alpha and beta for model3
    "answer_loss_coeff": 1,
    "span_loss_coeff": 1,
    "patience": 5,
    "checkpoint": 1000,
    "save_every": 1,
    "model_fname": "model.pt",
}


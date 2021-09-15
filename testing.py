from __future__ import unicode_literals, print_function, division

import sys

sys.path.append('/home/ubuntu/lab4')

# Reference
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# https://tw511.com/a/01/13026.html

# Import
from tools import *
# from tools import load_data_one_hot_encoded_with_sos_eos, *
import copy
import time
import torch
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter
from ptflops import get_model_complexity_info
from torch import nn, optim, TensorType
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchviz import make_dot
from io import open
import unicodedata
import string
import re
import random
import math
import torch
import os
from torch import optim
# import torch.nn.functional as F
# plt.switch_backend('agg')
import matplotlib.ticker as ticker
from os import system
from copy import deepcopy

# Preparation
os.chdir('/home/ubuntu/lab4')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training settings
MAX_LEN = 15
# KL_WEIGHT = 0.0
# BATCH_SIZ = 4

# Model settings
HIDDEN_SIZ = 512
COND_SIZ = 8
LATENT_SIZ = 32

# Fixed settings
LABEL_SIZ = 1
INPUT_SIZ = 1
SEQ_LEN = 1
N_CLASSES = 29
N_LABELS = 4
SOS_TOKEN = 0
EOS_TOKEN = 27
PAD_TOKEN = 28
assert LABEL_SIZ == SEQ_LEN == INPUT_SIZ == 1


# Note
# Tenses: simple present(sp), third person(tp), present progressive(pg), simple past(p).
# TODO: try one-hot & embedding
# Load training data
# Each row is a set of tenses of a word. [0] = sp; [1] = tp; [2] = pg; [3] = p
# Training data: 1227 rows x 4 tenses
# Every char's code: 0 = SOS; 1~26 = 'a'~'z'; 27 = EOS; 28 = PAD
# [(len(item[0]), item[0]) for family in DATA] and then .sort(): max word len = 15

# Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.label_embedding = nn.Embedding(N_LABELS, COND_SIZ)
        self.init_cond_fc = nn.Linear(HIDDEN_SIZ + COND_SIZ, HIDDEN_SIZ)
        self.input_embedding = nn.Embedding(N_CLASSES, HIDDEN_SIZ)
        self.lstm = nn.LSTM(input_size=HIDDEN_SIZ, hidden_size=HIDDEN_SIZ)

    def forward(self, x, hidden_state: torch.Tensor = None, cell_state=None, label=None):  # [input_siz]
        # Label & hidden state need to be given at the first time
        # Hidden state & cell state must be provided after first time
        batch_siz = x.shape[0]

        # Input
        x = self.input_embedding(x).view(SEQ_LEN, batch_siz, HIDDEN_SIZ)
        # Hidden state & cell state
        if label is not None:  # First time
            assert label is not None and hidden_state is not None
            label = self.label_embedding(label).view(batch_siz, COND_SIZ)
            hidden_state = torch.cat((hidden_state.view(batch_siz, HIDDEN_SIZ), label), dim=1)
            hidden_state = self.init_cond_fc(hidden_state).view(SEQ_LEN, batch_siz, HIDDEN_SIZ)
            _, (hidden_state, cell_state) = self.lstm(x, (hidden_state, hidden_state))
        else:
            assert hidden_state is not None and cell_state is not None
            _, (hidden_state, cell_state) = self.lstm(x, (hidden_state, cell_state))
        assert hidden_state is not None and cell_state is not None
        return hidden_state, cell_state

    @staticmethod
    def init_hidden(batch_siz):
        return torch.zeros(SEQ_LEN, batch_siz, HIDDEN_SIZ)


# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.hidden_mu_fc = nn.Linear(HIDDEN_SIZ, LATENT_SIZ)
        self.cell_mu_fc = nn.Linear(HIDDEN_SIZ, LATENT_SIZ)
        self.hidden_log_var_fc = nn.Linear(HIDDEN_SIZ, LATENT_SIZ)
        self.cell_log_var_fc = nn.Linear(HIDDEN_SIZ, LATENT_SIZ)
        self.hidden_label_embedding = nn.Embedding(N_LABELS, COND_SIZ)
        self.cell_label_embedding = nn.Embedding(N_LABELS, COND_SIZ)
        self.hidden_init_cond_fc = nn.Linear(LATENT_SIZ + COND_SIZ, HIDDEN_SIZ)
        self.cell_init_cond_fc = nn.Linear(LATENT_SIZ + COND_SIZ, HIDDEN_SIZ)
        self.input_embedding = nn.Embedding(N_CLASSES, HIDDEN_SIZ)
        self.lstm = nn.LSTM(input_size=HIDDEN_SIZ, hidden_size=HIDDEN_SIZ)
        self.output_fc = nn.Linear(HIDDEN_SIZ, N_CLASSES)

    def forward(self, x, hidden_state=None, cell_state=None, label=None, return_kl_loss=False):
        # Hidden state & cell state from encoder & label must be given at the first time
        # After the first time, hidden_state & cell_state must be provided
        hidden_reparameterization_return = None
        cell_reparameterization_return = None

        batch_siz = x.shape[0]

        # Input
        x = self.input_embedding(x).view(SEQ_LEN, batch_siz, HIDDEN_SIZ)

        # Hidden state & cell state
        if label is not None:  # First time
            # print('hidden_stttt',hidden_state)
            hidden_reparameterization_return = self.reparameterization(hidden_state, self.hidden_mu_fc,
                                                                       self.hidden_log_var_fc,
                                                                       return_loss=return_kl_loss, batch_siz=batch_siz)
            cell_reparameterization_return = self.reparameterization(cell_state, self.cell_mu_fc, self.cell_log_var_fc,
                                                                     return_loss=return_kl_loss, batch_siz=batch_siz)
            hidden_latent = hidden_reparameterization_return[0]
            cell_latent = cell_reparameterization_return[0]
            hidden_label = self.hidden_label_embedding(label).view(batch_siz, COND_SIZ)
            cell_label = self.cell_label_embedding(label).view(batch_siz, COND_SIZ)
            hidden_state = torch.cat((hidden_latent, hidden_label), dim=1)
            hidden_state = self.hidden_init_cond_fc(hidden_state).view(SEQ_LEN, batch_siz, HIDDEN_SIZ)
            cell_state = torch.cat((cell_latent, cell_label), dim=1)
            cell_state = self.hidden_init_cond_fc(cell_state).view(SEQ_LEN, batch_siz, HIDDEN_SIZ)
            outputs, (hidden_state, cell_state) = self.lstm(x, (hidden_state, cell_state))
        else:  # Others
            outputs, (hidden_state, cell_state) = self.lstm(x, (hidden_state, cell_state))
        outputs = self.output_fc(outputs.view(batch_siz, HIDDEN_SIZ))
        if return_kl_loss:
            return outputs, hidden_state, cell_state, hidden_reparameterization_return[1] + \
                   cell_reparameterization_return[1]
        else:
            return outputs, hidden_state, cell_state

    @staticmethod
    def reparameterization(encoder_state, mu_fc, log_var_fc, batch_siz, return_loss=False):
        encoder_state = encoder_state.view(batch_siz, HIDDEN_SIZ)
        mu = mu_fc(encoder_state)
        log_var = log_var_fc(encoder_state)
        epsilon = torch.randn_like(mu).to(DEVICE)
        latent = mu + torch.exp(log_var * 0.5) * epsilon
        if return_loss:
            loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - torch.exp(torch.clamp(log_var, max=10.)))  # KL Loss
            if torch.isinf(loss) or torch.isnan(loss):
                print('====================')
                print('kl_loss =', loss)
                print(encoder_state)
                print('log_var', log_var)
                print('mu', mu)
                print('====================')
                # loss = 0.0 * torch.sum(mu)  # Arbitrary
                raise ValueError('KLD loss explosion')
            return latent, loss
        else:
            return latent


DATA = load_data_with_word2noseq_and_eos_and_paired_by_label()


# def batch_encoder_forward(word_seqs: List[List[int]], labels: List[int], encoder):
#     label_tensor = torch.tensor([label], device=DEVICE)
#     encoder_hidden = encoder.init_hidden().to(DEVICE)
#     encoder_cell = None
#     word_len = len(word_seq)
#     # Encoder
#     for ei in range(word_len):
#         input_tensor = torch.tensor([word_seq[ei]], device=DEVICE)
#         encoder_hidden, encoder_cell = encoder(x=input_tensor,
#                                                hidden_state=encoder_hidden,
#                                                cell_state=None if ei == 0 else encoder_cell,
#                                                label=label_tensor if ei == 0 else None)
#     return encoder_hidden

# TODO: decoder lstm 吐出“］”或teacher force給出“EOS“都要break
# TODO: decoder 在算正確答案的word_len時要把第一個“］”後的都不算
def make_same_length_word_seqs(word_seqs: List[List[int]], add_sos=False, add_eos=False):
    for i in range(len(word_seqs)):
        if add_sos:
            word_seqs[i] = [SOS_TOKEN] + word_seqs[i]
        if add_eos:
            word_seqs[i] += [EOS_TOKEN]
    lengths = [len(word_seq) for word_seq in word_seqs]
    max_len = max(lengths)
    # print('max_len', max_len, [len(word) for word in word_seqs])
    # Fill "EOS" if not long enough
    result_seqs = []
    for i in range(len(word_seqs)):
        diff = max_len - lengths[i]
        # print('diff', diff, [EOS_TOKEN for _ in range(diff)])
        # word_seqs[i] += [EOS_TOKEN for _ in range(diff)]
        result_seqs.append(word_seqs[i] + [PAD_TOKEN for _ in range(diff)])
    # print('~~~~~~~~~~', [len(word) for word in result_seqs])
    return result_seqs


def encoder_forward(word_seqs: List[List[int]], labels: List[int], encoder):
    batch_siz = len(word_seqs)
    batch_label_tensor = torch.tensor(labels, device=DEVICE).view(batch_siz, LABEL_SIZ)
    batch_encoder_hidden = encoder.init_hidden(batch_siz=len(word_seqs)).to(DEVICE)
    batch_encoder_cell = None
    word_seqs = make_same_length_word_seqs(word_seqs)
    word_len = len(word_seqs[0])
    # Encoder
    for ei in range(word_len):
        input_tensor = torch.tensor(word_seqs, device=DEVICE).transpose(0, 1)[ei].view(batch_siz, INPUT_SIZ)
        batch_encoder_hidden, batch_encoder_cell = encoder(x=input_tensor,
                                                           hidden_state=batch_encoder_hidden,
                                                           cell_state=None if ei == 0 else batch_encoder_cell,
                                                           label=batch_label_tensor if ei == 0 else None)
    return batch_encoder_hidden, batch_encoder_cell


def decoder_forward(decoder, batch_encoder_hidden: torch.Tensor, batch_encoder_cell: torch.Tensor,
                    word_seqs: List[List[int]],
                    labels: List[int],
                    teacher_forcing_ratio, return_loss=False,
                    return_word_str=False, stop_when_eos=False, max_len_when_only_stop_in_eos=MAX_LEN):
    batch_siz = len(word_seqs)
    word_seqs = make_same_length_word_seqs(word_seqs, add_eos=True)

    batch_label_tensor = torch.tensor(labels, device=DEVICE).view(batch_siz, LABEL_SIZ)
    batch_decoder_input = torch.tensor([SOS_TOKEN for _ in range(batch_siz)], device=DEVICE).view(batch_siz, INPUT_SIZ)
    batch_decoder_hidden = batch_encoder_hidden
    batch_decoder_cell = batch_encoder_cell
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    cross_entropy = nn.CrossEntropyLoss()
    cross_entropy_loss = 0
    kl_loss = 0
    outputs = []
    whether_batch_eos = [False for _ in range(batch_siz)]
    word_len = len(word_seqs[0])
    for di in range(max_len_when_only_stop_in_eos if stop_when_eos else word_len):
        batch_decoder_return = decoder(x=batch_decoder_input,
                                       hidden_state=batch_decoder_hidden,
                                       cell_state=batch_decoder_cell,
                                       label=batch_label_tensor if di == 0 else None,
                                       return_kl_loss=True if di == 0 else False)
        batch_decoder_output = batch_decoder_return[0]
        batch_decoder_hidden = batch_decoder_return[1]
        batch_decoder_cell = batch_decoder_return[2]
        if di == 0:
            kl_loss = batch_decoder_return[3]

        if return_loss and di < word_len:  # 超過長度的部分不計
            # try:
            for b_i in range(batch_siz):
                truth = torch.tensor(word_seqs, device=DEVICE).transpose(0, 1)[di][b_i]
                if truth != PAD_TOKEN:
                    cross_entropy_loss += cross_entropy(batch_decoder_output[b_i].view(1, -1), truth.view(1))
            # print('batch_decoder_output', batch_decoder_output)
            # cross_entropy_loss += cross_entropy(batch_decoder_output,
            #                                    torch.tensor(word_seqs, device=DEVICE).transpose(0, 1)[di])
            # except:
            #     pass
            #     # print(batch_decoder_output)
            #     # print(torch.tensor(word_seqs, device=DEVICE).transpose(0, 1)[di])
            #     # raise ValueError('??????')
        if use_teacher_forcing:
            if di >= word_len:  # 超過預估長度
                raise NotImplementedError()
                # batch_decoder_input = torch.tensor([EOS_TOKEN], device=DEVICE)  # Teacher forcing
            else:
                batch_decoder_input = torch.tensor(word_seqs, device=DEVICE).transpose(0, 1)[di]  # Teacher forcing
        else:
            batch_decoder_input = torch.tensor(batch_decoder_output.argmax(dim=1), device=DEVICE).view(batch_siz,
                                                                                                       INPUT_SIZ)
        batch_output_char_int = batch_decoder_output.argmax(dim=1).view(batch_siz, INPUT_SIZ).tolist()
        if return_word_str:
            outputs.append([no2char(output_char_int[0]) for output_char_int in batch_output_char_int])
        if stop_when_eos:
            for item_i, item in enumerate(batch_output_char_int):
                if item[0] == EOS_TOKEN:
                    whether_batch_eos[item_i] = True
            if all(whether_batch_eos):
                break
                # if stop_when_eos and (output_char_int == EOS_TOKEN or decoder_input == EOS_TOKEN):
        #     if return_loss:
        #         for idx in range(di + 1, word_len):
        #             decoder_output = torch.tensor(one_hot_encode_char(EOS_TOKEN), device=DEVICE).view(1, -1)
        #             cross_entropy_loss += cross_entropy(decoder_output,
        #                                                 torch.tensor([word_seq[idx]], device=DEVICE)) * (
        #                                       1 if di <= 0.5 * word_len else 1)
        #     break
    ret = []
    if return_loss:
        ret.append((cross_entropy_loss, kl_loss))
    if return_word_str:
        ret.append([''.join([batch[i] for batch in outputs]) for i in range(batch_siz)])
    return tuple(ret)


def word_conversion(encoder, decoder, input_word_seq: List[int], input_label: int, output_label: int):
    #
    encoder = encoder.eval()
    decoder = decoder.eval()
    #
    batch_encoder_hidden, batch_encoder_cell = encoder_forward(deepcopy([input_word_seq]), [input_label], encoder)
    #
    output_str = \
        decoder_forward(decoder=decoder, batch_encoder_hidden=batch_encoder_hidden,
                        batch_encoder_cell=batch_encoder_cell, labels=[output_label],
                        word_seqs=deepcopy([input_word_seq]),
                        teacher_forcing_ratio=0, return_loss=False, return_word_str=True, stop_when_eos=True)[
            0]
    #
    # print('({}) {} -> ({}) {}'.format(input_label, noseq2word(input_word_seq), output_label, output_str))
    return output_str[0]


def tense_conversion_testing(encoder, decoder):
    test_data = [  # ([1, 2, 1, 14, 4, 15, 14], 0, 0, 'abandon]'),
        ([1, 2, 1, 14, 4, 15, 14], 0, 3, 'abandoned]'),
        ([1, 2, 5, 20], 0, 2, 'abetting]'),
        ([2, 5, 7, 9, 14], 0, 1, 'begins]'),
        ([5, 24, 16, 5, 14, 4], 0, 1, 'expends]'),
        ([19, 5, 14, 20], 3, 1, 'sends]'),
        ([19, 16, 12, 9, 20], 0, 2, 'splitting]'),
        ([6, 12, 1, 18, 5, 4], 3, 0, 'flare]'),
        ([6, 21, 14, 3, 20, 9, 15, 14, 9, 14, 7], 2, 0, 'function]'),
        ([6, 21, 14, 3, 20, 9, 15, 14, 9, 14, 7], 2, 3, 'functioned]'),
        ([8, 5, 1, 12, 9, 14, 7], 2, 1, 'heals]')  # , ([16, 18, 5, 4, 9, 3, 20], 0, 4, 'predicted]')
    ]
    total_bleu = 0
    for d in test_data:
        output_str = word_conversion(encoder, decoder, d[0], d[1], d[2])
        # print('output_str & d[3] & score:', output_str, d[3], compute_bleu_str(output_str, d[3]))
        total_bleu += compute_bleu_str(output_str, d[3])
        print('Input: {}\t Target: {}\t Prediction: {}'.format(noseq2word(d[0]), d[3], output_str))
    avg_tcab = total_bleu / len(test_data)
    print('Average BLEU score:', avg_tcab)
    return avg_tcab


N_GENERATE = 100
NOISES = load_rand_noises('/home/ubuntu/lab4/tensors', n=N_GENERATE, device='cuda')


def generation_testing(decoder):
    batch_siz = 1
    #
    decoder = decoder.eval()
    # Generate
    words = []
    for i in range(len(NOISES)):
        # batch_encoder_hidden = NOISES[i][0]
        # batch_encoder_cell = NOISES[i][1]
        batch_encoder_hidden = torch.randn((batch_siz, HIDDEN_SIZ), device=DEVICE)
        batch_encoder_cell = torch.randn((batch_siz, HIDDEN_SIZ), device=DEVICE)
        #
        family = []
        for tense_i in range(4):
            output_str = \
                decoder_forward(decoder=decoder, batch_encoder_hidden=deepcopy(batch_encoder_hidden),
                                batch_encoder_cell=deepcopy(batch_encoder_cell), labels=[tense_i],
                                word_seqs=deepcopy([[]]), teacher_forcing_ratio=0, return_loss=False,
                                return_word_str=True, stop_when_eos=True)[0]
            family.append(output_str[0].split(']')[0])
        words.append(family)
    # print(words)
    print('Gaussian score:', gaussian_score(words, n_print=100))


if __name__ == '__main__':
    # tc_encoder = torch.load('/home/ubuntu/lab4/saved_models/08-16-UTC19-03-tcab0.9027832524825772/encoder.pt')
    # print('Encoder loaded!')
    # tc_decoder = torch.load('/home/ubuntu/lab4/saved_models/08-16-UTC19-03-tcab0.9027832524825772/decoder.pt')
    # print('Decoder loaded!')
    times = 100
    # print('Avg tense conversion BLEU score:',
    #       sum([tense_conversion_testing(tc_encoder, tc_decoder) for _ in range(times)]) / float(times))
    # k = input()

    # encoder2 = torch.load('/home/ubuntu/lab4/saved_models/08-16-UTC20-43-tcab0.9332771455177624/encoder.pt')
    # decoder2 = torch.load('/home/ubuntu/lab4/saved_models/08-16-UTC20-43-tcab0.9332771455177624/decoder.pt')
    # print('en/decoder2 loaded!')
    # print('Avg tense conversion BLEU score:',
    #      sum([tense_conversion_testing(encoder2, decoder2) for _ in range(times)]) / float(times))
    #
    # generation_test_decoder = torch.load('/home/ubuntu/lab4/saved_good_gen/08-17-UTC00-12-gs0.22/decoder.pt')

    generation_test_decoder = torch.load('/home/ubuntu/lab4/saved_good_gen/08-17-UTC00-38-gs0.93/decoder.pt')
    print('Generation decoder loaded!')
    generation_testing(generation_test_decoder)

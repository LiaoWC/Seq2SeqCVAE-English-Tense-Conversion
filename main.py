from __future__ import unicode_literals, print_function, division
import sys

sys.path.append('/home/ubuntu/lab4')

# Reference
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# https://tw511.com/a/01/13026.html

# tcab = tense conversion average BLEU-score

# Import
from tools import *
from testing import generation_testing
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
            # loss = -0.5 * torch.sum(                1 + log_var - torch.clamp(mu.pow(2), max=10.) - torch.exp(torch.clamp(log_var, max=10.)))  # KL Loss
            loss = -0.5 * torch.sum(
                1 + log_var - torch.clamp(mu.pow(2), max=100.) - torch.exp(torch.clamp(log_var, max=10.)))  # KL Loss
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


def train(word_seqs: List[List[int]], labels: List[int], encoder, decoder, encoder_optimizer, decoder_optimizer,
          teacher_forcing_ratio, kl_weight):
    # Device
    encoder = encoder.to(DEVICE)
    decoder = decoder.to(DEVICE)
    # Init
    loss = torch.tensor(0., device=DEVICE)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Encoder
    batch_encoder_hidden, batch_encoder_cell = encoder_forward(deepcopy(word_seqs), labels, encoder)

    # Decoder
    ((cross_entropy_loss, kl_loss), output_word_str) = decoder_forward(decoder=decoder,
                                                                       batch_encoder_hidden=batch_encoder_hidden,
                                                                       batch_encoder_cell=batch_encoder_cell,
                                                                       word_seqs=deepcopy(word_seqs),
                                                                       labels=labels,
                                                                       teacher_forcing_ratio=teacher_forcing_ratio,
                                                                       return_loss=True, return_word_str=True,
                                                                       stop_when_eos=False)

    #
    loss += kl_loss * kl_weight + cross_entropy_loss
    # print('kl_loss', kl_loss, 'kl_w', kl_weight, 'ce_loss', cross_entropy_loss)
    loss /= len(word_seqs)
    if any(torch.isnan(loss.view(1))):
        print(word_seqs)
        print('kl_loss', kl_loss, 'kl_w', kl_weight, 'ce_loss', cross_entropy_loss)
        print([noseq2word(w) for w in word_seqs])
        print('Loss NAN!!!! but ignored')
        loss = -1
    # print('one train loss', loss)
    return (loss, cross_entropy_loss, kl_loss, kl_weight), [compute_bleu_str(output_str, noseq2word(word_seq)) for
                                                            (output_str, word_seq) in
                                                            zip(output_word_str, word_seqs)]


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


N_GENERATE = 100
NOISES = load_rand_noises('/home/ubuntu/lab4/tensors', n=N_GENERATE, device='cuda')


def generation_testing(decoder):
    # batch_siz = 1
    #
    decoder = decoder.eval()
    # Generate
    words = []
    for i in range(len(NOISES)):
        batch_encoder_hidden = NOISES[i][0]
        batch_encoder_cell = NOISES[i][1]
        # batch_encoder_hidden = torch.randn((batch_siz, HIDDEN_SIZ), device=DEVICE)
        # batch_encoder_cell = torch.randn((batch_siz, HIDDEN_SIZ), device=DEVICE)
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
    return gaussian_score(words)


def train_iters(data, encoder, decoder, n_iters, batch_siz, print_every, plot_every, teacher_force_ratio,
                learning_rate, kl_weight_mode: str):
    start = time.time()
    #
    print_loss_total = 0  # Reset every print_every
    print_ce_loss_total = 0
    print_kl_loss_total = 0
    print_bleu_total = 0
    print_tcab_total = 0
    train_time_used_total = 0.
    # Record
    loss_record = []
    ce_loss_record = []
    kl_loss_record = []
    bleu_record = []
    kl_weight_record = []
    teacher_force_record = []
    tcab_record = []
    gaussian_score_record = []
    #
    cur_best_bleu = 0
    cur_best_tcab = 0
    cur_best_gen_score = 0

    checkpoint_encoder = None
    checkpoint_decoder = None

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate, weight_decay=0.0001)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate, weight_decay=0.0001)

    print(f'KLD Weight Mode: {kl_weight_mode}')
    # training_samples = [random.choice(DATA) for _ in range(n_iters)]
    # all_samples = deepcopy(data)
    #
    # random.shuffle(all_samples)
    #
    # start_idx = 0

    for iteration in range(1, n_iters + 1):
        # for iteration in range(1 + 25000, n_iters + 1 + 25000):
        # KLD weight

        # s1 = time.time()

        if kl_weight_mode == 'monotonic':
            assert n_iters == 20000
            if iteration <= 10000:
                kl_weight = iteration / 10000.
            else:
                kl_weight = 1.0
        elif kl_weight_mode == 'cyclical':
            assert n_iters == 20000
            iter_indicator = iteration % 10000
            if iter_indicator <= 5000:
                kl_weight = iter_indicator / 5000.
            else:
                kl_weight = 1.0
        else:
            raise ValueError('Invalid kl weight mode:', kl_weight_mode)
        # if iteration <= 10000 // batch_siz:
        #     kl_weight = 0.05
        # elif 20000 // batch_siz < iteration <= 30000 // batch_siz:
        #     kl_weight = (iteration - 20000 // batch_siz) / (10000 // batch_siz) * 0.1
        # else:
        #     kl_weight = 0.2
        # pass
        # # kl_weight = 0.05
        # kl_weight += 0.1

        encoder.train()
        decoder.train()

        chosen = [random.choice(data) for _ in range(batch_siz)]

        word_seqs = [item[0] for item in chosen]
        labels = [item[1] for item in chosen]
        # word_seq, label = training_samples[iteration - 1]

        train_start_time = time.time()

        # print('s1:', time.time() - s1)
        # s2 = time.time()
        (batch_avg_loss, batch_ce_loss, batch_kl_loss, batch_kl_weight), batch_bleu_score_sum = train(
            word_seqs=word_seqs, labels=labels, encoder=encoder,
            decoder=decoder,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            teacher_forcing_ratio=teacher_force_ratio, kl_weight=kl_weight)

        train_time_used_total += time.time() - train_start_time
        # print('s2:', time.time() - s2)
        # s3 = time.time()
        if batch_avg_loss < 0:
            iteration -= 1
            continue
        batch_avg_loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        # Accumulate
        print_loss_total += batch_avg_loss.item() / len(word_seqs[0])
        print_bleu_total += sum(batch_bleu_score_sum) / batch_siz

        # print_ce_loss_total += batch_ce_loss.item() / len(word_seqs[0])
        # print_kl_loss_total += batch_kl_loss.item() / len(word_seqs[0])
        print_ce_loss_total += batch_ce_loss.item()
        print_kl_loss_total += batch_kl_loss.item()
        # print('kll', print_ce_loss_total, print_kl_loss_total)

        # Record
        bleu_record.append(sum(batch_bleu_score_sum) / batch_siz)
        loss_record.append(batch_avg_loss.item() / len(word_seqs[0]))
        kl_weight_record.append(kl_weight)
        # kl_loss_record.append(batch_kl_loss)
        # ce_loss_record.append(batch_ce_loss)
        teacher_force_record.append(teacher_force_ratio)
        # print('s3:', time.time() - s3)
        if iteration % print_every == 0:
            # s4 = time.time()

            # abandon 0 -> abandoned 3
            # abet 0 -> abetting 2
            # begin 0 -> begins 1
            # expend 0 -> expends 1
            # sent 3 -> sends 1
            # split 0 -> splitting 2
            # flared 3 -> flare 0
            # functioning 2 -> function 0
            # functioning 2 -> functioned 3
            # healing 2 -> heals 1
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
                ([8, 5, 1, 12, 9, 14, 7], 2, 1, 'heals]')]
            total_bleu = 0
            for d in test_data:
                output_str = word_conversion(encoder, decoder, d[0], d[1], d[2])
                # print('output_str & d[3] & score:', output_str, d[3], compute_bleu_str(output_str, d[3]))
                total_bleu += compute_bleu_str(output_str, d[3])
                # print('Input: {}\t Target: {}\t Prediction: {}'.format(noseq2word(d[0]), d[3], output_str))
            avg_tcab = total_bleu / len(test_data)
            print_tcab_total += avg_tcab
            # print(f' (TCAB {avg_tcab:.4f})')
            # Save models
            SAVE_MODEL = True
            if not SAVE_MODEL:
                print('WARNING: SAVE_MODEL = False')
            if avg_tcab > cur_best_tcab and SAVE_MODEL:
                print(f'(TCAB {avg_tcab:.4f}) in iter {iteration}')
                cur_best_tcab = avg_tcab
                save_dir = '/home/ubuntu/lab4/saved_models/' + datetime.now().strftime(
                    '%m-%d-UTC%H-%M') + f'-tcab{avg_tcab}'
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                torch.save(encoder, os.path.join(save_dir, 'encoder.pt'))
                torch.save(decoder, os.path.join(save_dir, 'decoder.pt'))
                # torch.save({'model': Encoder(), 'state_dict': encoder.state_dict(),
                #             'optimizer': encoder_optimizer.state_dict()}, os.path.join(save_dir, 'encoder.pt'))
                # torch.save({'model': Decoder(), 'state_dict': decoder.state_dict(),
                #             'optimizer': decoder_optimizer.state_dict()}, os.path.join(save_dir, 'decoder.pt'))
            # print('s4:', time.time() - s4)
        # plot_loss_total += loss
        if iteration % print_every == 0:
            # s5 = time.time()
            print_loss_avg = print_loss_total / print_every
            print_bleu_avg = print_bleu_total / print_every
            print_ce_loss_avg = print_ce_loss_total / print_every
            print_kl_loss_avg = print_kl_loss_total / print_every
            # tcab_record.append(print_tcab_total / print_every)
            tcab_record.append(print_tcab_total)
            ce_loss_record.append(print_ce_loss_avg)
            kl_loss_record.append(print_kl_loss_avg)
            print_loss_total = 0
            print_bleu_total = 0
            print_tcab_total = 0
            print_ce_loss_total = 0
            print_kl_loss_total = 0
            print(f'Time used for training: {train_time_used_total} sec')
            train_time_used_total = 0
            # print('\n\nTraining input:', noseq2word(word_seq), label)
            print('%s (%d %d%%) (loss %.4f) (bleu %.4f) (klw %.3f)' % (
                time_since(start, iteration / n_iters), iteration, iteration / n_iters * 100, print_loss_avg,
                print_bleu_avg, kl_weight))
            #
            # # # Let's see "abandon" verb
            test_word_seq = [1, 2, 1, 14, 4, 15, 14]
            test_label = 0  # sp
            res_word_0 = word_conversion(encoder, decoder, test_word_seq, test_label, 0)
            res_word_1 = word_conversion(encoder, decoder, test_word_seq, test_label, 1)
            res_word_2 = word_conversion(encoder, decoder, test_word_seq, test_label, 2)
            res_word_3 = word_conversion(encoder, decoder, test_word_seq, test_label, 3)
            print(' abandon -> {}, {}, {}, {}'.format(res_word_0, res_word_1, res_word_2, res_word_3))
            # if print_bleu_avg > cur_best_bleu or checkpoint_encoder is None:
            #     cur_best_bleu = print_bleu_avg
            #     print('--------------------- Replace!! ---------------------')
            #     checkpoint_encoder = deepcopy(encoder)
            #     checkpoint_decoder = deepcopy(decoder)
            # else:
            #     encoder = checkpoint_encoder
            #     decoder = checkpoint_decoder

            SAVE_GEN_MODEL = True
            if not SAVE_GEN_MODEL:
                print('WARNING: SAVE__GEN_MODEL = False')
            gen_score = generation_testing(decoder)
            if gen_score > 0.00 and SAVE_GEN_MODEL:
                print(f'(gen_score {gen_score:.4f}) in iter {iteration}')
                cur_best_gen_score = gen_score
                save_dir = '/home/ubuntu/lab4/saved_good_gen/' + datetime.now().strftime(
                    '%m-%d-UTC%H-%M') + f'-gs{cur_best_gen_score}'
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                torch.save(encoder, os.path.join(save_dir, 'encoder.pt'))
                torch.save(decoder, os.path.join(save_dir, 'decoder.pt'))
            print('Current gaussian score ------------> ', gen_score)
            gaussian_score_record.append(gen_score)
            # print('s5:', time.time() - s5)

        if iteration % plot_every == 0:
            # s6 = time.time()
            assert plot_every % print_every == 0
            print('========== PLOTTING =========')
            fig, ax1 = plt.subplots(figsize=(12, 7.2))
            ax2 = ax1.twinx()
            # plt.plot(np.arange(len(loss_record)) + 1, loss_record, label='loss')
            x_by_print_every = [(i + 1) * print_every for i in range(len(tcab_record))]
            ax1.plot(x_by_print_every, kl_loss_record, label=f'KLD Loss (avg of {print_every} iters)', color='C0')
            ax1.plot(x_by_print_every, ce_loss_record, label=f'CE Loss (avg of {print_every} iters)', color='C1')
            ax1.set_xlabel(f'Iteration (batch size = {batch_siz})')
            ax1.set_ylabel('Loss')
            ax1.legend()

            # plt.plot(np.arange(len(bleu_record)) + 1, bleu_record, label='training_bleu')

            ax2.set_ylabel('Score/Weight')
            ax2.scatter(x_by_print_every, tcab_record, label='Tense Conversion BLEU', color='C2')
            ax2.plot(np.arange(len(kl_weight_record)) + 1, kl_weight_record, label='KLD Weight', color='C3',
                     linestyle='dashed')
            ax2.plot(np.arange(len(teacher_force_record)) + 1, teacher_force_record, label='Teacher Ratio', color='C4',
                     linestyle='dashed')
            ax2.scatter(x_by_print_every, gaussian_score_record, label='Gaussian Score', color='C5')
            ax2.legend()
            plt.title(f'Training Loss/Ratio Curve')

            plt.savefig(os.path.join('/home/ubuntu/lab4/plot/',
                                     datetime.now().strftime(
                                         '%m-%d-UTC%H-%M') +
                                     f'-bs{batch_siz}-iter{iteration}-lr{learning_rate}-KLMode{kl_weight_mode}.png'))
            plt.show()
            # print('s6:', time.time() - s6)

        # Save each iter
        save_dir = '/home/ubuntu/lab4/saved_each_iter/' + datetime.now().strftime(
            '%m-%d-UTC%H-%M')
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(encoder, os.path.join(save_dir, 'encoder.pt'))
        torch.save(decoder, os.path.join(save_dir, 'decoder.pt'))


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
        ([8, 5, 1, 12, 9, 14, 7], 2, 1, 'heals]')]
    total_bleu = 0
    for d in test_data:
        output_str = word_conversion(encoder, decoder, d[0], d[1], d[2])
        # print('output_str & d[3] & score:', output_str, d[3], compute_bleu_str(output_str, d[3]))
        total_bleu += compute_bleu_str(output_str, d[3])
        print('Input: {}\t Target: {}\t Prediction: {}'.format(noseq2word(d[0]), d[3], output_str))
    avg_tcab = total_bleu / len(test_data)
    print('Average BLEU score:', avg_tcab)


model_encoder = Encoder()
model_decoder = Decoder()
# model_encoder3 = Encoder()
# model_decoder3 = Decoder()

# Trained with 0.1 kl_weight and still has good performance on TCAB
# /home/ubuntu/lab4/saved_models/08-16-UTC18-36-tcab0.6385088125974092/encoder.pt
# /home/ubuntu/lab4/saved_models/08-16-UTC18-36-tcab0.6385088125974092/decoder.pt

if __name__ == '__main__':
    # loaded_encoder = torch.load('/home/ubuntu/lab4/saved_models/08-16-UTC19-22-tcab0.9168240945554477/encoder.pt')
    # loaded_decoder = torch.load('/home/ubuntu/lab4/saved_models/08-16-UTC19-22-tcab0.9168240945554477/decoder.pt')
    train_iters(data=deepcopy(DATA),
                encoder=model_encoder,  # loaded_encoder,
                decoder=model_decoder,  # loaded_decoder,
                batch_siz=1,
                # n_iters=30000,
                n_iters=20000,
                print_every=500,
                plot_every=2000,
                teacher_force_ratio=1,
                learning_rate=0.001,
                # kl_weight_mode='monotonic')
                kl_weight_mode='cyclical')

    # test_word = 'revise'
    # test_word_seq = word2noseq(test_word) + [27]
    # test_label = 0  # sp
    # res_word_0 = word_conversion(model_encoder, model_decoder, test_word_seq, test_label, 0, model_mu_fc, model_log_var_fc)
    # res_word_1 = word_conversion(model_encoder, model_decoder, test_word_seq, test_label, 1, model_mu_fc, model_log_var_fc)
    # res_word_2 = word_conversion(model_encoder, model_decoder, test_word_seq, test_label, 2, model_mu_fc, model_log_var_fc)
    # res_word_3 = word_conversion(model_encoder, model_decoder, test_word_seq, test_label, 3, model_mu_fc, model_log_var_fc)
    # print(' {}] -> {}, {}, {}, {}'.format(test_word, res_word_0, res_word_1, res_word_2, res_word_3))

from typing import Union, List, Tuple
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import time
import math
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

TRAINING_DATA_PATH = '/home/ubuntu/lab4/dataset/train.txt'


# TRAINING_DATA_PATH = '/home/ubuntu/lab4/dataset/try.txt'


def load_data():
    return [line.split(' ') for line in Path(TRAINING_DATA_PATH).read_text().split('\n')]


# Number of word families
def print_number_of_word_families():
    print('Number of word families:', len(load_data()))


# Check char frequencies
def char_freq():
    all_chars = ''.join([''.join(word_family) for word_family in load_data()])
    counter = Counter(all_chars)
    counter = sorted([[c, counter[c]] for c in counter], key=lambda x: x[1], reverse=True)
    plt.title('Frequencies of All Characters')
    plt.bar([c[0] for c in counter], [c[1] for c in counter])
    plt.show()


#


#
def one_hot_encode_char(char: Union[str, int]) -> List:
    """ One-hot encode a single char. [0] = SOS; [1]~[26] = 'a'~'z'; [27] = EOS
    :param char: single-char str or int. If int: 0 is SOS; 27 is EOS; 1~26 are A~Z
    :return: 1D ndarray
    """
    if isinstance(char, str):
        char = ord(char) - ord('a') + 1
    char_vector = np.zeros((28,))
    char_vector[char] = 1.
    return char_vector.tolist()


def one_hot_encode_word_with_sos_eos(word: str) -> List[List]:
    """ One-hot encode a word with SOS and EOS placed at the front and end respectively.
    :param word: string that all of its characters are alphabets.
    :return: 2D ndarray
    """
    encoded_chars = [one_hot_encode_char(c) for c in word]
    encoded_chars.insert(0, one_hot_encode_char(0))
    encoded_chars.append(one_hot_encode_char(27))
    return encoded_chars


def load_data_one_hot_encoded_with_sos_eos() -> List[List[List[List]]]:
    """
    :return: word family -> word in different tenses -> a word with chars -> char vector
    """
    encoded = []
    for word_family in load_data():
        encoded.append([one_hot_encode_word_with_sos_eos(word) for word in word_family])
    return encoded


def char_vector_to_char(char_vec: List):
    """ Return the meaning of a char."""
    char_num = np.argmax(char_vec, axis=0)
    if char_num == 0:
        return '['  # 'SOS'
    elif char_num == 27:
        return ']'  # 'EOS'
    elif 1 <= char_num <= 26:
        return chr(ord('a') - 1 + char_num)
    else:
        raise ValueError('Invalid char number:', char_num)


def word_vector_to_word(word_vec: List[List]):
    """ Return the meaning of a word."""
    ans = ''
    for char in word_vec:
        ans += char_vector_to_char(char)
    return ans


def char2no(char: str) -> int:
    no = ord(char[0]) - ord('a') + 1
    if no > 26 or no < 1:
        raise ValueError('Invalid char:', char)
    return no


def word2noseq(word: str) -> List[int]:
    return list(char2no(c) for c in word)


def noseq_add_eos(noseq: List[int]) -> List[int]:
    return noseq + [27]


def no2char(no: int) -> str:
    if no == 0:
        return '['
    elif no == 27:
        return ']'
    elif no == 28:
        return '*'
    elif 1 <= no <= 26:
        return chr(ord('a') + no - 1)
    else:
        raise ValueError('Invalid number:', no)


def noseq2word(noseq: List[int]) -> str:
    return ''.join(no2char(no) for no in noseq)


def load_data_with_word2noseq_and_eos() -> List[List[List[int]]]:
    return [[word2noseq(word) for word in family_words] for family_words in load_data()]


def load_data_with_word2noseq_and_eos_and_paired_by_label() -> List[Tuple[List[int], int]]:
    data = load_data_with_word2noseq_and_eos()
    data_with_label = []
    for word_family in data:
        assert len(word_family) == 4
        for i in range(len(word_family)):
            data_with_label.append((word_family[i], i))
    return data_with_label


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def remove_after_first_eos(word: str):
    return word[:word.find(']')]


# compute BLEU-4 score
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33, 0.33, 0.33)
    else:
        weights = (0.25, 0.25, 0.25, 0.25)
    return sentence_bleu([reference], output, weights=weights, smoothing_function=cc.method1)


def compute_bleu_str(str1, str2):
    return compute_bleu([c for c in remove_after_first_eos(str1)], [c for c in remove_after_first_eos(str2)])


def gaussian_score(words, n_print=5):
    words_list = []
    score = 0
    with open(TRAINING_DATA_PATH, 'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        # print('============================ Ans ============================')
        # for family in sorted(words_list):
        #     print(family)
        print('============================ Gen ============================')
        for i, family in enumerate(sorted(words)):
            print(family)
            if i >= n_print:
                break
        for t in words:
            for i in words_list:
                if t == i:
                    score += 1
    return score / len(words)


def generate_noises(save_dir: str, n: int, hidden_size: int):
    import os
    for i in range(1, n + 1):
        torch.save(torch.randn((1, hidden_size)), os.path.join(save_dir, f'{i}_hidden_state.pt'))
        torch.save(torch.randn((1, hidden_size)), os.path.join(save_dir, f'{i}_cell_state.pt'))


# generate_noises('/home/ubuntu/lab4/tensors', n=100, hidden_size=512)

def load_rand_noises(save_dir: str, n, device):
    import os
    filenames = []
    root_name = ''
    for root, dirs, files in os.walk(save_dir):
        root_name = root
        for file in files:
            filenames.append(os.path.join(root, file))
    filenames.sort()
    assert len(filenames) % 2 == 0
    tensor_pairs = []
    for i in range(n):
        h = torch.load(os.path.join(root_name, f'{i + 1}_hidden_state.pt')).to(device)
        c = torch.load(os.path.join(root_name, f'{i + 1}_cell_state.pt')).to(device)
        tensor_pairs.append((h, c))
    return tensor_pairs

# noises = load_rand_noises('/home/ubuntu/lab4/tensors', n=100, device='cuda')

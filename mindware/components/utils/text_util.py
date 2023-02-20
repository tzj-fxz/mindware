# -*- coding: UTF-8 -*-

import numpy as np
import jieba
import logging
import os
import tqdm
import pickle as pkl
from mindware.components.feature_engineering.transformation_graph import DataNode
from mindware.datasets.base_dl_dataset import TextDataset, TotalTextDataset
jieba.setLogLevel(logging.INFO)

import sys, importlib
importlib.reload(sys)


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")  # 防止编码错误
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass
    def reset(self):
        self.log.close()
        sys.stdout=self.terminal


maxlen = 100
MAX_VOCAB_SIZE = 50000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def content2Xy(data):
    ret_data_X = []
    ret_data_y = []
    for slice in data:
        ret_data_X.append(slice[0])
        ret_data_y.append(slice[1])
    return ret_data_X, ret_data_y


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    # print(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    # print(vocab_dic)
    return vocab_dic


def build_dataset(vocab_path, train_path, val_path, test_path, use_word):
    if use_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    vocab = build_vocab(train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
    pkl.dump(vocab, open(vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=1000):
        # TODO 
        # pad_size 改为 torchtext 的处理
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in f:
                lin = line.strip()
                if not lin:
                    continue
                if len(lin.split('\t')) == 2:
                    content, label = lin.split('\t')
                else:
                    content = lin.split('\t')[0]
                    label = 0
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]

    train = load_dataset(train_path)
    dev = load_dataset(val_path)
    test = load_dataset(test_path)
    return vocab, train, dev, test



def get_stopwords(path=r"/root/tzj/dataset/工单分类验证数据/stopwords.txt"):
    f = open(path, "r", encoding="utf-8")
    stopwords = f.read().split('\n')
    return stopwords


# get whole X dataset, cut words, then return new X dataset
# used in initializing
def cut_tokens(lines):
    stopword_list = get_stopwords()
    token_lines = []
    num_line = 0
    print(len(lines))
    for line in lines:
        tokens_this_line = jieba.lcut(line)
        effective_tokens = []
        for token in tokens_this_line:
            if token in stopword_list:
                continue
            effective_tokens.append(token)
        token_lines.append(effective_tokens)
        num_line += 1
        if num_line % 100 == 0:
            print(num_line)
    return token_lines


# generate token dict from both trainset and testset, save and return it
def get_token_dict(token_lines_from_train, path):
    stopword_list = get_stopwords()
    token_dict = {}
    for token_line in token_lines_from_train:
        for token in token_line:
            if token in stopword_list:
                continue
            if token in token_dict:
                token_dict[token] += 1
            else:
                token_dict[token] = 1
    token_list = sorted(token_dict.items(), key=lambda item: item[1], reverse=True)
    # save token list
    token_idx_dict = {}
    f = open(path, "w", encoding="utf-8")
    for idx, token in enumerate(token_list):
        entry = token[0] + " " + str(idx+1) + " " + str(token[1]) + "\n"
        token_idx_dict[token[0]] = idx+1
        if not os.path.exists(path):
            f.write(entry)
    return token_idx_dict


# translate tokens to "index" vectors, then return it 
def token_to_vector(token_lines, total_dataset: TotalTextDataset):
    token_idx_dict = total_dataset.train_token_dict
    
    stopword_list = get_stopwords()
    index_lines = []
    length_lines = []
    for token_line in token_lines:
        # print(token_line)
        index_line = []
        for word in token_line:
            if word in stopword_list:
                continue
            if word not in token_idx_dict.keys():
                index_line.append(0)
            else:
                index_line.append(int(token_idx_dict[word]))
        if len(index_line) > maxlen:
            index_line = index_line[:maxlen]
        else:
            index_line.extend([0] * (maxlen-len(index_line)))
        index_lines.append(index_line)
        length_lines.append(len(index_line))
    # print(sorted(length_lines))
    return index_lines


def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    if lower:
        text = text.lower()

    translate_dict = dict((c, split) for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]


def build_embeddings_index(glove_path='./glove_data/glove.6B.50d.txt'):
    embeddings_index = dict()
    f = open(glove_path, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def load_embedding_matrix(word_index, embedding_index, if_normalize=True):
    all_embs = np.stack(embedding_index.values())
    if if_normalize:
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
    else:
        emb_mean, emb_std = 0, 1
    embed_size = all_embs.shape[1]
    nb_words = len(word_index)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i - 1] = embedding_vector
    return embedding_matrix


def load_text_embeddings(texts, embedding_index, method='average', alpha=1e-3):
    from keras.preprocessing.text import Tokenizer
    tok = Tokenizer()
    tok.fit_on_texts(texts)
    word_index = tok.word_index
    embedding_matrix = load_embedding_matrix(word_index, embedding_index, if_normalize=False)
    index_sequences = tok.texts_to_sequences(texts)
    text_embeddings = []
    if method == 'average':
        for seq in index_sequences:
            embedding = []
            for i in seq:
                embedding.append(embedding_matrix[i - 1])
            text_embeddings.append(np.average(embedding, axis=0))
        text_embeddings = np.array(text_embeddings)
    elif method == 'weighted':
        from collections import Counter
        for seq in index_sequences:
            counter = Counter(seq)
            embedding = []
            for i in counter:
                embedding.append(embedding_matrix[i - 1] * alpha / (alpha + counter[i] / len(seq)))
            text_embeddings.append(np.average(embedding, axis=0))
        text_embeddings = np.array(text_embeddings)

        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
        svd.fit(text_embeddings)
        pc = svd.components_
        test_embeddings = text_embeddings - text_embeddings.dot(pc.transpose()) * pc
    return text_embeddings
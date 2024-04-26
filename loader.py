import torch
from torch.utils.data import Dataset
import numpy as np
import jieba
import codecs
import re
from collections import Counter
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def read_file(filename):
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z]+)")  # the method of cutting text by punctuation
    contents, labels = [], []
    with codecs.open(filename,'r',encoding='utf-8') as f:
        for line in f:
            try:
                line=line.rstrip()
                assert len(line.split('\t'))==2
                label,content=line.split('\t')
                labels.append(label)
                blocks = re_han.split(content)
                word = []
                for blk in blocks:
                    if re_han.match(blk):
                        for w in jieba.cut(blk):
                            if len(w) >= 2:
                                word.append(w)
                contents.append(word)
            except:
                pass
    return labels, contents


def build_vocab(filenames, vocab_dir, vocab_size=8000):
    all_data = []
    for filename in filenames:
        _, contents = read_file(filename)
        for content in contents:
            all_data.extend(content)
    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size-1)
    words, _ = list(zip(*count_pairs))
    words = ['<PAD>']+list(words)

    with codecs.open(vocab_dir, 'w', encoding='utf-8') as f:
        f.write('\n'.join(words)+'\n')


def read_vocab(vocab_dir):
    words = codecs.open(vocab_dir, 'r', encoding='utf-8').read().strip().split('\n')
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


def process_file_pytorch(filename, word_to_id, cat_to_id, max_length=600):
    labels, contents = read_file(filename)
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append(torch.tensor([word_to_id[x] for x in contents[i] if x in word_to_id]))
        label_id.append(cat_to_id[labels[i]])

    x_pad = pad_sequence(data_id, batch_first=True, padding_value=0)  # 填充值0通常代表<PAD>
    if x_pad.size(1) > max_length:
        x_pad = x_pad[:, :max_length]  # 截断超过最大长度的部分

    y_tensor = torch.tensor(label_id)

    return x_pad, y_tensor


def batch_iter(x_pad, y_pad, batch_size=64):
    data_len = len(x_pad)
    num_batch = int((data_len-1)/batch_size)+1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x_pad[indices]
    y_shuffle = y_pad[indices]

    for i in range(num_batch):
        start_id = i*batch_size
        end_id = min((i+1)*batch_size,data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def export_word2vec_vectors(vocab, word2vec_dir, trimmed_filename):
    file_r = codecs.open(word2vec_dir, 'r', encoding='utf-8')
    line = file_r.readline()
    voc_size, vec_dim = map(int, line.split(' '))
    embeddings = np.zeros([len(vocab), vec_dim])
    line = file_r.readline()
    while line:
        try:
            items = line.split(' ')
            word = items[0]
            vec = np.asarray(items[1:], dtype='float32')
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(vec)
        except:
            pass
        line = file_r.readline()
    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_training_word2vec_vectors(filename):
    with np.load(filename) as data:
        return data["embeddings"]

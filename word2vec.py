import logging
import time
import codecs
import re
import jieba
from torch.utils.data import Dataset
from gensim.models import word2vec
from model import TextConfig

re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z]+)")


class Get_sentences(Dataset):
    def __init__(self, filenames):
        self.sentences = []
        for filename in filenames:
            with codecs.open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        line = line.strip()
                        line = line.split('\t')
                        assert len(line) == 2
                        blocks = re_han.split(line[1])
                        words = []
                        for blk in blocks:
                            if re_han.match(blk):
                                words.extend(jieba.lcut(blk))
                        self.sentences.append(words)
                    except:
                        continue

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


def train_word2vec(filenames):
    t1 = time.time()
    dataset = Get_sentences(filenames)
    all_sentences = [words for words in dataset]
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", level=logging.INFO)

    model = word2vec.Word2Vec(min_count=1, vector_size=100, window=5, workers=6)
    model.build_vocab(corpus_iterable=all_sentences)
    model.train(corpus_iterable=all_sentences, total_examples=model.corpus_count, epochs=5)

    model.wv.save_word2vec_format(config.vector_word_filename, binary=False)
    print('-------------------------------------------')
    print("Training word2vec model cost %.3f seconds...\n" % (time.time() - t1))


if __name__ == '__main__':
    config = TextConfig()
    filenames = [config.train_filename, config.test_filename, config.val_filename]
    train_word2vec(filenames)
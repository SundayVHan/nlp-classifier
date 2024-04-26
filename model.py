import torch
import torch.nn as nn
import torch.nn.functional as F


class TextConfig():
    embedding_size = 100
    vocab_size = 8000
    pre_training = None

    seq_length = 600
    num_classes = 10

    num_filters = 128
    filter_sizes = [2, 3, 4]

    keep_prob = 0.5
    lr = 1e-3
    lr_decay = 0.9
    clip = 6.0
    l2_reg_lambda = 0.01

    num_epochs = 10
    batch_size = 64
    print_per_batch = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_filename = './data/cnews.train.txt'
    test_filename = './data/cnews.test.txt'
    val_filename = './data/cnews.val.txt'
    vocab_filename = './data/vocab.txt'
    vector_word_filename = './data/vector_word.txt'
    vector_word_npz = './data/vector_word.npz'


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size)
        self.embedding.weight.data.copy_(torch.from_numpy(config.pre_training))

        self.conv = nn.ModuleList([
            nn.Conv2d(1, config.num_filters, (k, config.embedding_size)) for k in config.filter_sizes
        ])
        self.dropout = nn.Dropout(config.keep_prob)
        self.fc = nn.Linear(len(config.filter_sizes) * config.num_filters, config.num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
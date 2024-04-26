import os.path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from model import TextConfig
from model import TextCNN
from loader import *
import time


def evaluate(model, data_loader, criterion, config):
    model.eval()
    total_loss = 0.0
    corrects = 0
    total_examples = 0
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(config.device), y_batch.to(config.device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            corrects += (torch.max(outputs, 1)[1].view(y_batch.size()).data == y_batch.data).sum()
            total_examples += y_batch.size(0)

    return total_loss / len(data_loader), corrects.float() / total_examples


def train_model():
    tensorboard_dir = './tensorboard/textcnn'
    save_dir = './checkpoint/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'model.pt')

    print("Loading training data...")
    start_time = time.time()
    x_train, y_train = process_file_pytorch(config.train_filename, word_to_id, cat_to_id, config.seq_length)
    x_val, y_b = process_file_pytorch(config.val_filename, word_to_id, cat_to_id, config.seq_length)
    print("Time cost: %.3f seconds...\n" % (time.time() - start_time))

    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataset = TensorDataset(torch.tensor(x_val, dtype=torch.long), torch.tensor(y_b, dtype=torch.long))
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    writer = SummaryWriter(tensorboard_dir)

    print('Training and evaluating...')
    best_val_accuracy = 0
    last_improved = 0
    require_improvement = 1000
    global_step = 0
    flag = False

    for epoch in range(config.num_epochs):
        model.train()
        print('Epoch', epoch + 1)
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(config.device), y_batch.to(config.device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            if global_step % config.print_per_batch == 0:
                corrects = (torch.max(output, 1)[1].view(y_batch.size()).data == y_batch.data).sum()
                train_accuracy = corrects / float(y_batch.size(0))
                val_loss, val_accuracy = evaluate(model, val_loader, criterion, config)
                writer.add_scalar('loss/train', loss.item(), global_step)
                writer.add_scalar('accuracy/train', train_accuracy, global_step)

                if val_accuracy > best_val_accuracy:
                    torch.save(model.state_dict(), save_path)
                    best_val_accuracy = val_accuracy
                    last_improved = global_step
                    improved_str = '*'
                else:
                    improved_str = ''
                print("step: {}, train loss: {:.3f}, train accuracy: {:.3f}, val loss: {:.3f}, val accuracy: {:.3f}, training speed: {:.3f} sec/batch {}\n".format(
                    global_step, loss.item(), train_accuracy, val_loss, val_accuracy,
                    time.time() - start_time, improved_str))
                start_time = time.time()

            global_step += 1
            if global_step - last_improved > require_improvement:
                print("No optimization over 1000 steps, stop training")
                flag = True
                break

        if flag:
            break
        config.lr *= config.lr_decay


if __name__ == '__main__':
    config = TextConfig()
    filenames = [config.val_filename, config.train_filename, config.test_filename]
    if not os.path.exists(config.vocab_filename):
        build_vocab(filenames, config.vocab_filename)

    categories, cat_to_id = read_category()
    words,word_to_id = read_vocab(config.vocab_filename)
    config.vocab_size = len(words)

    if not os.path.exists(config.vector_word_npz):
        export_word2vec_vectors(word_to_id, config.vector_word_filename, config.vector_word_npz)
    config.pre_training = get_training_word2vec_vectors(config.vector_word_npz)

    model = TextCNN(config)
    train_model()

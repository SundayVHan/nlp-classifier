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
import sklearn.metrics as metrics


def evaluate(model, data_loader):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(config.device), y_batch.to(config.device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            acc = (outputs.argmax(1) == y_batch).float().mean()
            total_loss += loss.item() * x_batch.size(0)
            total_acc += acc.item() * x_batch.size(0)

    return total_loss / len(data_loader.dataset), total_acc / len(data_loader.dataset)


def test_model():
    t1 = time.time()
    print("Loading test data...")
    x_test, y_test = process_file_pytorch(config.test_filename, word_to_id, cat_to_id, config.seq_length)
    test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    print('Testing...')
    test_loss, test_accuracy = evaluate(model, test_loader)
    print('Test Loss: {:.6f}, Test Acc: {:.2%}'.format(test_loss, test_accuracy))

    y_test_cls = y_test.numpy()
    y_pred_cls = np.array([])

    model.eval()
    with torch.no_grad():
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(config.device)
            outputs = model(x_batch)
            y_pred_cls = np.append(y_pred_cls, outputs.argmax(1).cpu().numpy())

    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    print("Time usage: {:.3f} seconds".format(time.time() - t1))


if __name__ == '__main__':
    config = TextConfig()

    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(config.vocab_filename)
    config.vocab_size = len(words)

    if not os.path.exists(config.vector_word_npz):
        export_word2vec_vectors(word_to_id, config.vector_word_filename, config.vector_word_npz)
    config.pre_training = get_training_word2vec_vectors(config.vector_word_npz)
    model = TextCNN(config)

    # Load saved model weights
    save_path = './checkpoint/textcnn/model.pt'
    model.load_state_dict(torch.load(save_path))

    test_model()
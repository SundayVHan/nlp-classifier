import torch
from model import *
from loader import *


config = TextConfig
config.pre_training = get_training_word2vec_vectors(config.vector_word_npz)
model = TextCNN(config)
save_path = './checkpoint/textcnn/model.pt'
model.load_state_dict(torch.load(save_path))
model.eval()

categories, cat_to_id = read_category()
words,word_to_id = read_vocab(config.vocab_filename)
config.vocab_size = len(words)


def preprocess(content):
    contents = []
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z]+)")
    blocks = re_han.split(content)
    words = []
    for blk in blocks:
        if re_han.match(blk):
            for w in jieba.cut(blk):
                if len(w) >= 2:
                    words.append(w)
    indices = [word_to_id[word] for word in words if word in word_to_id]
    contents.append(indices)
    ret = torch.tensor(contents, dtype=torch.long)
    return ret


input_tensor = preprocess("禅师：科比出场时间将增加 能赢球他打40分钟也无妨新浪体育讯北京时间4月28日消息，即便科比-布莱恩特现在是带伤出战，但由于洛杉矶湖人队将面临一场晋级战，主帅菲尔-杰克逊明确表示会增加科比的出场时间，保罗-加索尔则不愿用伤病来作为自己发挥不佳的借口。")


with torch.no_grad():
    output = model(input_tensor)
    predicted_prob, predicted_index = torch.max(output, 1)
    predicted_class = categories[predicted_index.item()]

print("Predicted class:", predicted_class)
print("Probability:", predicted_prob.item())

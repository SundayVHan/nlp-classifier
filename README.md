# nlp-classifier
基于pytorch实现的情感自然语言处理nlp模型，实现对较长评论的文本分类<br>
具体模型为TextCNN模型，复现自gaussic的论文《text-classification-cnn-rnn》<br>
<br>
相较于原论文，本模型采用jieba对中文句子进行单词划分，并以中文词汇为token进行训练与预测<br>
<br>
本模型仅用作课程作业，模型效果一般

# 环境
1. python3.11
2. 详情参考requirements.txt

# 数据集
[THUCTC：一个高效的中文文本分类工具包](http://thuctc.thunlp.org/)<br>
<br>
也可以使用百度云盘下载<br>
[百度云盘](https://pan.baidu.com/s/1B1SxVuRulSZTbUml-aQHWA?pwd=awds) 验证码:awds<br>
<br>
下载完成后放入工程根目录下的data文件夹中（chinese-classifier/data/{下载的文件})

# 使用
1. 训练词向量 python word2vec.py
2. 训练模型 python train.py
3. 测试模型 python test.py
4. 自定义验证模型 predict.py p.s.输入的句子请在源码中自行修改，不要太短（否则CNN的卷积核大小会超过句子分词的数量）
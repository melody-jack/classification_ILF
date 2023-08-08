import os
import codecs
import numpy as np
import math
from tensorflow.keras.utils import to_categorical
from keras_bert import load_vocabulary, load_trained_model_from_checkpoint, Tokenizer, get_checkpoint_paths,gen_batch_inputs
from keras_bert.bert import TOKEN_CLS, TOKEN_SEP
from keras_bert.datasets import get_pretrained, PretrainedList
import logging as log
log.basicConfig(level=log.INFO)
tag_file = os.path.join(os.path.dirname(__file__), 'hface','tags')
label2id = {'其他':0,
            '赌博':1,
            '传销':2,
            }
#自定义Token工具，对原有的规则做补充
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R=[]
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')
            else:
                R.append('[UNK]')
        return R

def bulid_tokenizer(dict_path):
    token_dict = {}
    with codecs.open(dict_path,'r','utf8') as read:
        for line in read:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return OurTokenizer(token_dict)
def read_corpus(data_file):
    with codecs.open(data_file, 'r', encoding='utf8') as f:
        data = f.readlines()

    return data




class DataGenerator:
    """
    训练和测试用数据的生成器
    """
    def __init__(self, data, tokenizer, label2id, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = int(math.ceil(len(self.data) / self.batch_size))
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):
        return self.steps

    def seq_padding(self, X, padding=0):
        """
        对句向量X中所有token进行长度统计，取最长值。
        之后使用padding(0)补足到最长值
        param: X: 语料list
        param: padding: 填充值
        """
        # 每个句子的token长度
        L = [len(x) for x in X]
        # 最大句子token长度
        ML = max(L)
        if ML > 512:
            log.error('max length:%r'%ML)
        return np.array([ np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])


    def __iter__(self):
        batch_token_idx, batch_segment_idx, batch_labels = [], [], []
        while True:
            # 随机提取语料
            # np.random.shuffle(self.data)

            for i, data in enumerate(self.data):
                if len(data.split()) != 1:

                    line, label = data.strip().split('\t')

                    if len(line) > 510:
                        # log.error("too longer:%r \n 长度:%r" % (line, len(line)))
                        continue
                    # bert 模型需要的token索引和label索引
                    token_idx = [self.tokenizer._token_dict[TOKEN_CLS]]
                    # 对语料进行逐字符编码
                    for s in line:
                        s_token_idx = self.tokenizer.encode(s)[0][1:-1]
                        token_idx.extend(s_token_idx)
                    token_idx.extend([self.tokenizer._token_dict[TOKEN_SEP]])
                    segment_idx = [0] * len(token_idx)
                    batch_token_idx.append(token_idx)
                    batch_segment_idx.append(segment_idx)
                    batch_labels.append([label2id[label]])

                    # 如果数量达到了训练的批次，将数据补齐后交由生成器调度
                    if len(batch_token_idx) == self.batch_size or i == len(self.data) - 1:
                        batch_token_idx = self.seq_padding(batch_token_idx)
                        batch_segment_idx = self.seq_padding(batch_segment_idx)
                        # batch_labels = self.seq_padding(batch_labels)
                        yield [batch_token_idx, batch_segment_idx], to_categorical(batch_labels,num_classes=len(label2id))
                        batch_token_idx, batch_segment_idx, batch_labels = [], [], []
                else:
                    continue

    def forfit(self):
        while True:
            for d in self.__iter__():
                yield d
if __name__=="__main__":
    # 加载中文预训练模型(缓存在当前用户.keras/datasets目录中)
    model_path = get_pretrained(PretrainedList.chinese_base)
    # 模型所在目录的path
    paths = get_checkpoint_paths(model_path)

    # 加载token字典
    token_dict = load_vocabulary(paths.vocab)
    # 创建tokenizer
    tokenizer = Tokenizer(token_dict)
    # 训练和验证集语料
    data_file=os.path.join(os.path.dirname(__file__),'save_corpus.txt')
    data = read_corpus(data_file)

    # 模型训练用生成器
    datas = DataGenerator(data, tokenizer,label2id, batch_size=4).__iter__()

    (x1, x2), y = next(datas)
    log.info(x1)
    log.info(x2)
    log.info(y)
    # print(datas.steps)

    # for i, data in enumerate(data):
    #     if len(data.split()) != 1:
    #         line, label = data.strip().split('\t')
    #     else:
    #         continue
    #     print(label)
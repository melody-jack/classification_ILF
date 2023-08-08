from tensorflow.keras.models import load_model
from keras_bert import get_custom_objects
from tensorflow.python.keras.layers.core import Lambda
from corpus_process import *
from handle_corpus import *

# 加载中文预训练模型(缓存在当前用户.keras/datasets目录中)
model_path = get_pretrained(PretrainedList.chinese_base)
# 模型所在目录的path
paths = get_checkpoint_paths(model_path)
#model save path
model_file = os.path.join(os.path.dirname(__file__),'models','bert_class.h5')

#加载token字典
token_dict = load_vocabulary(paths.vocab)
# 创建tokenizer
tokenizer = Tokenizer(token_dict)

#数据对齐
def seq_padding(X, padding=0):
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

def reload_model():
    """模型加载"""
    custom_objs = get_custom_objects()
    my_objs = {'lambda':Lambda(lambda x:x[:,0])}
    custom_objs.update(my_objs)
    model = load_model(model_file,custom_objects=custom_objs)

    return model

def predict(model, text):
    input_seq = tokenizer.encode(first=text)

    X1 = seq_padding([input_seq[0]])
    X2 = seq_padding([input_seq[1]])

    res = model.predict([X1,X2])

    id2label = {v: k for k, v in label2id.items()}
    return id2label[np.argmax(res)]


if __name__=="__main__":
    #加载模型
    model = reload_model()
    #预测
    text="'祝贺:《爸爸去哪儿》送出128000及奖品',请及时登录领用.验证码8890.详情:aszzwa.com.本次活动已通过互联网公证处审批,真实有效,请放心领用!"
    text1=dels(text)
    res=predict(model,text1)
    print(text1)
    print(res)



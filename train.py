from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model,load_model
from keras_bert import get_custom_objects
from tensorflow.python.keras.layers.core import Lambda
from tensorflow.keras import backend as K
from corpus_process import *
import tensorflow as tf
#使用gpu运行
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 加载中文预训练模型(缓存在当前用户.keras/datasets目录中)
model_path = get_pretrained(PretrainedList.chinese_base)
# 模型所在目录的path
paths = get_checkpoint_paths(model_path)
#加载预训练模型
bert_model = load_trained_model_from_checkpoint(paths.config,paths.checkpoint,trainable=True,seq_len=None)
#批次
batch_size=4
#轮数
epochs=5


#model save path
model_file = os.path.join(os.path.dirname(__file__),'models','bert_class.h5')

#precision
def Precision(y_true, y_pred):
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.20), 'float32'))
    pred_positives = K.sum(K.cast(K.greater(K.clip(y_pred, 0, 1), 0.20), 'float32'))

    precision = true_positives / (pred_positives + K.epsilon())
    return precision

#recall
def Recall(y_true, y_pred):
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.20), 'float32'))
    poss_positives = K.sum(K.cast(K.greater(K.clip(y_true, 0, 1), 0.20), 'float32'))

    recall = true_positives / (poss_positives + K.epsilon())
    return recall

# # f-measure
# def F1(y_true, y_pred):
#     p_val = Precision(y_true, y_pred)
#     r_val = Recall(y_true, y_pred)
#     f_val = 2*p_val*r_val / (p_val + r_val)
#     return f_val

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

def create_model():
    """
    创建bert文本分类模型
    """
    #模型输入为token_index和segment_index
    x1_input = Input(shape=(None,),dtype='int32')
    x2_input = Input(shape=(None,),dtype='int32')


    bert = bert_model([x1_input,x2_input])
    x = Lambda(lambda x:x[:,0])(bert) #取bert输出的第一个值CLS
    dro = Dropout(0.5)(x)
    den = Dense(128,activation='relu')(dro)
    dro = Dropout(0.3)(den)

    prob = Dense(3,activation='softmax')(dro)
    model = Model([x1_input,x2_input],prob)

    #模型编译
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, metrics=['acc', Precision, Recall], loss='categorical_crossentropy')
    model.summary()
    return model

def train_model(model,train_gen,valid_gen,epochs,callbacks=None):
    """模型训练"""
    model.fit_generator(
        train_gen,
        steps_per_epoch=int(math.ceil(len(train_data) / batch_size)),
        epochs=epochs,
        validation_data=valid_gen,
        validation_steps=int(math.ceil(len(val_data) / batch_size)),
        callbacks=callbacks)

def model_save(model):
    """模型保存"""
    model.save(model_file)

def reload_model():
    """模型加载"""
    custom_objs = get_custom_objects()
    my_objs = {'lambda':Lambda(lambda x:x[:,0])}
    custom_objs.update(my_objs)
    model = load_model(model_file,custom_objects=custom_objs)

    return model

if __name__=="__main__":
    #加载token字典
    token_dict = load_vocabulary(paths.vocab)
    # 创建tokenizer
    tokenizer = Tokenizer(token_dict)
    # 训练和验证集语料
    data_file = os.path.join(os.path.dirname(__file__), 'save_corpus.txt')
    data = read_corpus(data_file)

    #切分训练集与测试集
    num=len(data)
    split_rate=0.2
    train_data=data[int(num*split_rate):]
    val_data=data[:int(num*split_rate)]

    train_gen = DataGenerator(train_data,tokenizer,label2id,batch_size).__iter__()
    valid_gen = DataGenerator(val_data,tokenizer,label2id,batch_size).__iter__()

    #创建模型
    model = create_model()
    #模型训练
    train_model(model,train_gen,valid_gen,epochs,callbacks=None)

    model_save(model)





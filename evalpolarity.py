import numpy as np
import random
from passage.preprocessing import Tokenizer
from passage.layers import Embedding,GatedRecurrent,Dense
from passage.models import RNN
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score


def readData(file_text,file_label):
    pos_text = []
    neg_text = []
    data_text = open(file_text).read().split('\n')
    data_label = open(file_label).read().split('\n')
    #    print data_text, data_label
    num = 0
    for item in data_label:

        if item.strip() == "positive":
            pos_text.append(data_text[num])
        else:
            neg_text.append(data_text[num])
        num += 1
    print len(pos_text), len(neg_text)
    return pos_text,neg_text


def filesplit(pos_text,neg_text,train_pos_num,train_neg_num):
    train_label = [1] * train_pos_num
    train_label.extend([0] * train_neg_num)
    train_pos = random.sample(pos_text,train_pos_num)
    test_pos = list(set(pos_text).difference(set(train_pos)))
    train_neg = random.sample(neg_text,train_neg_num)
    test_neg = list(set(neg_text).difference(set(train_neg)))
    test_label = [1] * len(test_pos)
    test_label.extend([0] * len(test_neg))
    train_pos.extend(train_neg)
    test_pos.extend(test_neg)
    return train_pos,train_label,test_pos,test_label


def datasetsplit(pos_text,neg_text,train_pos_num,train_neg_num):
    train_text = []
    train_label = []
    test_text = []
    test_label = []
    num = 0
    for item in pos_text:
        if num < train_pos_num:
            train_text.append(item)
            train_label.append(1)
        else:
            test_text.append(item)
            test_label.append(1)
        num += 1
    num = 0
    for item in neg_text:
        if num < train_neg_num:
            train_text.append(item)
            train_label.append(0)
        else:
            test_text.append(item)
            test_label.append(0)
        num += 1
    return train_text,train_label,test_text,test_label


def metric(test_true,test_pred):
    test_true = np.array(test_true)
    pred_label = []
    for item in test_pred.tolist():
        if item[0] > 0.5:
            pred_label.append(1)
        else:
            pred_label.append(0)
    print "precision_score =",precision_score(test_true,np.array(pred_label))
    print "recall_score =",recall_score(test_true,np.array(pred_label))
    print "f1_score =",f1_score(test_true,np.array(pred_label))


def rnn(train_text,train_label):
    tokenizer = Tokenizer()
    train_tokens = tokenizer.fit_transform(train_text)
    layers = [
        Embedding(size=50,n_features=tokenizer.n_features),
        GatedRecurrent(size=128),
        Dense(size=1,activation='sigmoid')
    ]
    #    print "train_tokens=", train_tokens
    model = RNN(layers=layers,cost='BinaryCrossEntropy')
    model.fit(train_tokens,train_label)
    return model


def predict(model,test_text):
    tokenizer = Tokenizer()
    result = model.predict(tokenizer.fit_transform(test_text))
    #    print result.shape
    #    print "result =", result
    return result


if __name__ == "__main__":
    train_pos_num = 500
    train_neg_num = 100
    file_text = "text.txt"
    file_label = "polarity.txt"
    pos_text,neg_text = readData(file_text,file_label)
    print type(pos_text),type(neg_text)
    train_text,train_label,test_text,test_label = filesplit(pos_text,neg_text,train_pos_num,train_neg_num)
    print len(train_text),len(test_text)
    print "=========Model Start ============"
    model = rnn(train_text,train_label)
    print "=========Predict Start ============"
    predict_result = predict(model,test_text)
    metric(test_label,predict_result)
    print "=========Job Finish ============"

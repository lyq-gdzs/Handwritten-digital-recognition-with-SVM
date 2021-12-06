import gzip
import struct
import numpy as np
import _pickle as pickle
from sklearn.metrics import roc_curve, auc,confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle
import itertools

if __name__ == '__main__':
    with gzip.open('../MNIST_data/t10k-labels-idx1-ubyte.gz', 'rb') as tlpath:
        _, _ = struct.unpack('>II', tlpath.read(8))
        labels = np.fromstring(tlpath.read(), dtype=np.uint8)
    with gzip.open('../MNIST_data/t10k-images-idx3-ubyte.gz', 'rb') as tipath:
        _, _, rows, cols = struct.unpack('>IIII', tipath.read(16))
        images = np.fromstring(tipath.read(), dtype=np.uint8).reshape(len(labels), 784)
    # 自定义测试样本数
    num_test=1000
    labels_test=labels[0:num_test]
    # one-hot 编码
    labels_test = label_binarize(labels_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    images_test=images[0:num_test]
    # 归一化
    images_test=np.round(images_test/255)
    with open('./model_60k.pkl', 'rb') as file:
        model = pickle.load(file)
    # TN = np.full(10,num_test, int)
    # FN = np.zeros(10, int)
    # TP = np.zeros(10, int)
    # FP = np.zeros(10, int)

    # for i in range(num_test):
    #     if (z[i]!=labels_test[i]).any():
    #         FN[temp[i]]+=1 # 判为负，实际为正
    #         FP[z[i].tolist().index(1)] += 1 # 判为正，实际为负
    #     if (z[i]==labels_test[i]).all(): # 判为正，实际为正
    #         TP[temp[i]]+=1
    # TN=TN-FN-FP-TP # 判为负，实际为负
    # print(TN,'\n',FN,'\n',TP,'\n',FP)

    # target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8', 'class 9']
    # print(classification_report(labels_test, z, target_names=target_names))


    # 计算每一类别的ROC曲线
    q = model.decision_function(images_test)
    FPR = dict() # 真正例率
    TPR = dict() # 假正例率
    ROC_AUC = dict()
    for i in range(10):
        FPR[i], TPR[i], _ = roc_curve(labels_test[:,i],q[:,i])
        ROC_AUC[i] = auc(FPR[i], TPR[i])
    lw = 2 # 线条宽度
    plt.figure()

    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd','#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    # 并行遍历
    for i, color in zip(range(10), colors):
        plt.plot(FPR[i], TPR[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.3f})'
                       ''.format(i, ROC_AUC[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

    # 计算混淆矩阵
    q = np.argmax(q, axis=1)
    labels_test=np.argmax(labels_test,axis=1)
    print('准确率:', np.sum(q == labels_test) / q.size)
    cm = confusion_matrix(labels_test, q)
    plt.figure()
    # 指定分类类别
    classes = range(np.max(labels_test) + 1)
    title = 'Confusion matrix'
    # 混淆矩阵颜色风格
    cmap = plt.cm.jet
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    # 按照行和列填写百分比数据
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    #



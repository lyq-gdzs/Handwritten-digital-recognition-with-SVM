import gzip
import struct
import numpy as np
from sklearn import svm
import _pickle as pickle
import time
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
start=time.time()

with gzip.open('../MNIST_data/train-labels-idx1-ubyte.gz', 'rb') as lpath:
    #使用struct.unpack方法读取前两个数据，>代表高位在前，I代表32位整型。lpath.read(8)表示一次从文件中读取8个字节
    #这样读到的前两个数据分别是magic number和样本个数，对于本实验无用，故以'_'代替
    _, _ = struct.unpack('>II',lpath.read(8))
    #使用np.fromstring读取剩下的数据，lpath.read()表示读取所有的数据
    labels = np.fromstring(lpath.read(),dtype=np.uint8)
with gzip.open('../MNIST_data/train-images-idx3-ubyte.gz', 'rb') as ipath:
    _, _, rows, cols = struct.unpack('>IIII',ipath.read(16))
    images = np.fromstring(ipath.read(),dtype=np.uint8).reshape(len(labels), 784)

# 自定义训练样本数
num_train=60000
labels_train=labels[0:num_train]
# one-hot编码
labels_train = label_binarize(labels_train, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# 归一化
images_train=np.round(images[0:num_train]/255)
model = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=np.random.RandomState(0)))
model.fit(images_train,labels_train)
with open('./model_60k.pkl', 'wb') as file:
    pickle.dump(model, file)
end=time.time()
print('time cost:','%.3f'%(end-start),'s')


import numpy as np
import _pickle as pickle
import cv2
def JudgeEdge(img, length, flag, size):
    threshold = 80
    for i in range(length):
        #Cow or Column 判断是行是列
        if flag == 0:
            #Positive sequence 正序判断该行是否有手写数字
            line1 = img[i,:]
            #Negative sequence 倒序判断该行是否有手写数字
            line2 = img[length-1-i,:]
        else:
            line1 = img[:,i]
            line2 = img[:,length-1-i]
        #If edge, recode serial number 若有手写数字，即到达边界，记录下行
        if sum(line1)>=threshold and size[0]==-1:
            size[0] = i
        if sum(line2)>=threshold and size[1]==-1:
            size[1] = length-1-i
        #If get the both of edge, break 若上下边界都得到，则跳出
        if size[0]!=-1 and size[1]!=-1:
            break
    return size
def CutPicture(img):
    #初始化新大小
    size = []
    #图片的行数
    length = len(img)
    #图片的列数
    width = len(img[0,:])
    #计算新大小
    size.append(JudgeEdge(img, length, 0, [-1, -1]))
    size.append(JudgeEdge(img, width, 1, [-1, -1]))
    size = np.array(size).reshape(4)
    print(size)
    return img[size[0]:size[1]+1, size[2]:size[3]+1]
def img_preprocess(img_path):
    img_cv = cv2.imread(img_path)  # 读取数据
    img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # img = CutPicture(img)
    img = cv2.resize(img, (28, 28))
    # cv2.imshow('img', img)
    # cv2.waitKey(1)
    img_normalization = np.round(img / 255)  # 对灰度值进行归一化
    return np.reshape(img_normalization, (1, -1))  # 1 * 400 矩阵

if __name__ == '__main__':
    path=r'my_num\test_4_1.png'
    with open('.\model_60k.pkl', 'rb') as file:
        model = pickle.load(file)
    q = model.decision_function(img_preprocess(path))
    print(np.argmax(q))

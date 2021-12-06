
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import _pickle as pickle

import demo
import numpy as np

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog") # 创建实例，名为Dialog
        Dialog.resize(645, 475)
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(230, 340, 141, 41)) # 设置button位置
        self.pushButton.setAutoDefault(False)
        self.pushButton.setObjectName("pushButton")

        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(220, 50, 191, 221)) # "显示图片"label的位置
        self.label.setWordWrap(False)
        self.label.setObjectName("label")

        self.textEdit = QtWidgets.QTextEdit(Dialog)
        self.textEdit.setGeometry(QtCore.QRect(220, 280, 191, 41)) # 设置文本框大小
        self.textEdit.setObjectName("textEdit")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog) # 根据信号名称自动连接到槽函数的核心代码

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "识别"))
        self.pushButton.setText(_translate("Dialog", "打开图片"))
        self.label.setText(_translate("Dialog", "图片区域"))


class MyWindow(QMainWindow, Ui_Dialog):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.openImage)

    def openImage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "my_num")
        png = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height()) # 限制显示图片大小为label大小
        self.label.setPixmap(png) # 将图片铺到label中
        self.textEdit.setText(imgName) # 将图片名写入文本框

        with open('./model_60k.pkl', 'rb') as file:
            model = pickle.load(file)
        dataMat = demo.img_preprocess(imgName)
        preResult = model.decision_function(dataMat)

        self.textEdit.setReadOnly(True)
        self.textEdit.setStyleSheet("color:red")
        self.textEdit.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.textEdit.setFontPointSize(9)
        self.textEdit.setText("预测的结果是：")
        self.textEdit.append(str(np.argmax(preResult)))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())
    input()

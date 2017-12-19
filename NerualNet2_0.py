# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 09:58:13 2017

@author: Administrator
"""

import numpy as np
def logistic(x):
    return 1/(1+np.exp(-x))
def logistic_deriv(x):
    return logistic(x)*(1-logistic(x))
class NeuralNet:
    def __init__(self,layer_struct):
        self.weights=[]##a list of numpy array
        self.bios=[]
        self.activation=logistic
        self.activation_deriv=logistic_deriv
        for i in range(len(layer_struct)-1):
            weight=np.random.random((layer_struct[i+1],layer_struct[i]))-0.5
            bios=np.random.random((layer_struct[i+1],1))-0.5##列向量
            self.weights.append(weight)
            self.bios.append(bios)
    
    def backprop(self,trainsample,label):
        ##don't forget transpose   #trainsample是一维array  变成二维以后转置
        trainsample=np.atleast_2d(trainsample).T
        temp=np.zeros((self.bios[-1].shape[0],1))
        ##行向量
        temp[label,0]=1   ##label是类别标签，是一个数
        label=temp     ##把类别标签转换成向量，相当于onehot编码
        z=[1]##a list of numpy array
        a=[trainsample]##为了保持下标一致，z的第一项用1代替
        w_num=len(self.weights)   ##权值向量的个数，相当于节点层数减一
        for j in range(w_num):
                ##lie向量  W*x+b
                z.append(np.dot(self.weights[j],a[j])+self.bios[j])
                ##lie向量  a=sigmoid(z)
                a.append(self.activation(z[j+1]))
            ##正向传播完成
            ##C=np.sum((label-a[-1])**2/2)
            ##cross_entropy
        deltas=[1]*w_num##cost对每一蹭z的偏导 每一项初始化为1 实际是a list of numpy array
                ##单独计算输出层delta：cost对每一层的Z求导值，每一项对应相乘，得到行向量
                ##完全理解交叉熵函数求导
                ##交叉熵函数作为cost function 完胜输出值与标签的差值二次方
                ##差值二次方作为cost function时：
        ##deltas[-1]=(a[-1]-label)*self.activation_deriv(z[-1])
        deltas[-1]=a[-1]-label
                ##非输出层delta迭代计算
        for j in range(w_num-1,0,-1):
                    ##行向量,长度为该层节点数
                    ##数学推导
            deltas[j-1]=self.activation_deriv(z[j])*np.dot(self.weights[j].T,deltas[j])
        nabla_w=[]
        nabla_b=[]##进来一个样本，得到一个变化值，为后来求平均变化做准备
        for j in range(w_num):
                ##同型矩阵相加减
            nabla_w.append(np.dot(deltas[j],a[j].T))
            nabla_b.append(deltas[j])
        return nabla_w,nabla_b
    
    def update_mini_batch(self,mini_batch,eta):
        nabla_w_sum=[np.zeros(w.shape) for w in self.weights]
        nabla_b_sum=[np.zeros(b.shape) for b in self.bios]
                    ##进去一个样本，累加一个变化量，最后根据一个mini_batch_size更新w,b
        for trainsample,label in mini_batch:##mini_batch是一个列表，里面是多个元组，元祖里面是一个样本
            nabla_w,nabla_b=self.backprop(trainsample,label)
            nabla_w_sum=[nabla_w_eachlayer+nabla_w_sum_eachlayer for nabla_w_eachlayer,nabla_w_sum_eachlayer in zip(nabla_w,nabla_w_sum)]
            nabla_b_sum=[nabla_b_eachlayer+nabla_b_sum_eachlayer for nabla_b_eachlayer,nabla_b_sum_eachlayer in zip(nabla_b,nabla_b_sum)]
        self.weights=[w-eta*nw/len(mini_batch) for w,nw in zip(self.weights,nabla_w_sum)]
        self.bios=[b-eta*nb/len(mini_batch) for b,nb in zip(self.bios,nabla_b_sum)]
        
    def fit(self,trainmat,labelmat,learning_rate=0.1,mini_batch_size=10,cycles=100):
        print('正在训练神经网络，这可能需要较长时间，请等待。。。')
        for i in range(cycles):
            if i%100==0:
                print('第%d轮循环：'%i)
                      ##把训练数据与标签绑定
                      ##traindata是一维数组，label是一个数，合并成一个元组
                      ##traindata_label_set是一个元组组成的列表
            traindata_label_set=[(traindata,label) for traindata,label in zip(trainmat,labelmat)]
            m=len(trainmat)
                     ##打乱顺序
            np.random.shuffle(traindata_label_set)
                     ##把乱序后的数据集分化为多份，注意仍然是列表
                     ##mini_batches  形如[[(),(),...],
#                                         [(),(),...],
#                                         [],
#                                         ...]     每一个元组里面是一个样本和标签
            mini_batches=[traindata_label_set[k:k+mini_batch_size] for k in range(0,m,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,learning_rate)

            
        print('训练完成！')
        
    def predict(self,testmat):
        a=np.atleast_2d(testmat).T
        for i in range(len(self.weights)):
            z=np.dot(self.weights[i],a)+self.bios[i]
            a=self.activation(z)
        return a.argmax(axis=0)
                

if __name__=='__main__':
    a=np.random.randint(-20,20,(100,3))
    for each in a:
        if each[0]>0 and each[1]>0:
            each[2]=1
        elif each[0]<0 and each[1]>0:
            each[2]=2
        elif each[0]<0 and each[1]<0:
            each[2]=3
        elif each[0]>0 and each[1]<0:
            each[2]=4
        else:
            each[2]=0
    x_train=a[:,:2]
    y_train=a[:,2]
    b=np.random.randint(-20,20,(100,3))
    for each in b:
        if each[0]>0 and each[1]>0:
            each[2]=1
        elif each[0]<0 and each[1]>0:
            each[2]=2
        elif each[0]<0 and each[1]<0:
            each[2]=3
        elif each[0]>0 and each[1]<0:
            each[2]=4
        else:
            each[2]=0
    nn=NeuralNet([2,5,5])
    nn.fit(x_train,y_train)
    y_train_pred=nn.predict(x_train)
    
    x_test=b[:,:2]
    y_test=b[:,2]
    y_test_pred=nn.predict(x_test)
    print('训练集正确率',np.sum(y_train_pred==y_train)/len(y_train))
    print('测试集正确率',np.sum(y_test_pred==y_test)/len(y_test))
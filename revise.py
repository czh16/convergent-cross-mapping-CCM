# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 22:21:42 2020

@author: czh
"""

import numpy as np
from numpy import *
import xlrd
import matplotlib.pyplot as plt
book=xlrd.open_workbook('D:\pythonworkspace\CCM\ESM3_Data_moran.xlsx')     #获取数据
RhoSetCCM=[]
libsizeSet=[]
lib_total=1000  #读入的文件总长度
i_sample_set=[] #取样点，200个
e=2             ##嵌入维度，需要另外计算
y_tilde_lib_total_sample=[]   #用于计算相似度的，Y的流型的真实值，共计200个
for i_1 in np.linspace(0,lib_total-10,200):  #取样点
            i_sample_set.append(int(i_1))     #取Y的流型上的第0,4,9...等200个点
              
x_total=[]  #作为结果    
y_total=[]  #作为结果 

sheet=book.sheet_by_index(0)               
for n in range(1,lib_total):
        x_total.append(sheet.cell(n,3).value)   ##读取第二列的数据~~~~~~~~~~~~~~~~~~~~~~~~需要修改
        y_total.append(sheet.cell(n,4).value)   ##读取第二列的数据~~~~~~~~~~~~~~~~~~~~~~~~需要修改
lenth_total=len(x_total)-e+1                 #x全部流型上点的个数
x_tilde_lib_total = [0] * lenth_total        #x流型全部的x_tilde_lib_total
y_tilde_lib_total = [0] * lenth_total        #Y流型全部的x_tilde_lib_total                      
for i in range(len(x_total)-e+1):
        x_tilde_lib_total[i]=[x_total[i]]
        y_tilde_lib_total[i]=[y_total[i]]
        for i2 in range(e-1):
            x_tilde_lib_total[i].append(x_total[i2+i+1])
            y_tilde_lib_total[i].append(y_total[i2+i+1])

for i in i_sample_set:   
    y_tilde_lib_total_sample.append(y_tilde_lib_total[i])
#print(i_sample_set)       

def CCM(lib,column_x,column_y):       #通过x的值去预测y    Y——> X              
    ##print(column_y,"-->",column_x)
    x=[]  #作为结果    
    y=[]  #作为原因
    column_x=column_x
    column_y=column_y
    libsize=lib  #library size
    
    sheet=book.sheet_by_index(0)               
    for n in range(1,libsize):
        x.append(sheet.cell(n,column_x).value)   ##读取第二列的数据
        y.append(sheet.cell(n,column_y).value)   ##读取第三列的数据,下面产生上面

    y_hat=[]                        #通过CCM预测的y值
    
    lenth=len(x)-e+1                #部分流型上点的个数
    x_tilde = [0] * lenth           #X部分流型
    y_tilde = [0] * lenth           #Y部分流型

    for i in range(len(x)-e+1):
        x_tilde[i]=[x[i]]
        y_tilde[i]=[y[i]]
        for i2 in range(e-1):
            x_tilde[i].append(x[i2+i+1])       #x部分流型
            y_tilde[i].append(y[i2+i+1])       #y部分流型
    
    dist_x=np.mat(np.zeros((200,lenth)))  #X全部流型上200个取样点+X部分流型lenth个点的欧式距离矩阵
    for h in range(200):
        for w in range(lenth):
            dist_x[h,w] = np.linalg.norm(np.array(x_tilde[w]) - np.array(x_tilde_lib_total[i_sample_set[h]]))         
    
    dist_x_list=dist_x.tolist()            #X全部流型上+部分流型lenth个点的欧式距离矩阵
    #print(dist_x_list)
    dist_x_list_sort=dist_x_list.copy()    #排序好的距离矩阵dist_x_list_sort
    #print(dist_x_list_sort)
    t_x=[]                                #200个取样点的最近临近点的t,t是三维的
    for t_i in range(200):                #t_i表示距离矩阵的行数
        dist_x_list_sort[t_i]=sorted(dist_x_list[t_i])     ##距离矩阵按照行，进行排序，从小到大
        t1=dist_x_list[t_i].index(dist_x_list_sort[t_i][1])##后面的1表示除了0外，倒数第一小的数
        t2=dist_x_list[t_i].index(dist_x_list_sort[t_i][2])##后面的2表示除了0外，倒数第二小的数
        t3=dist_x_list[t_i].index(dist_x_list_sort[t_i][3])##后面的2表示除了0外，倒数第3小的数
        t_x.append([t1,t2,t3])  #取出e+1个最近点,这里e=2,因此取3个最近点,200x3的矩阵
    #print(t_x)
    for hat_i in range(200):#简化计算，将三个点的权重赋值为0.58，0.28,0.14
        result=np.array(y_tilde[t_x[hat_i][0]])*0.58+np.array(y_tilde[t_x[hat_i][1]])*0.28+np.array(y_tilde[t_x[hat_i][2]])*0.14
        y_hat.append(result)        #200个取样点的预测值y_hat，二维的
    #print(y_hat)
    #print(y_tilde_lib_total_sample)
    

    def CalculateRho(y_hat,y_tilde_lib_total_sample):  #200个取样点进行相似度计算
        temp1=[0] * e     #200个点的预测值、真实值都是二维的
        temp2=[0] * e      
        for i_mean in range(200):
            temp1=temp1+np.array(y_tilde_lib_total_sample[i_mean])
            temp2=temp2+np.array(y_hat[i_mean])
        y_tilde_mean=temp1/200     #真实的y的均值
        y_hat_mean=temp2/200       #预测的y的均值
        #print(y_tilde_mean)
        #print(y_hat_mean)
        result_1=0
        result_2=0
        result_3=0
        for i_1 in range(200): ##
            result_1=result_1+(np.linalg.norm(y_tilde_lib_total_sample[i_1]-y_tilde_mean))*(np.linalg.norm(y_hat[i_1]-y_hat_mean))
            result_2=result_2+(np.linalg.norm(y_tilde_lib_total_sample[i_1]-y_tilde_mean))**2
            result_3=result_3+(np.linalg.norm(y_hat[i_1]-y_hat_mean))**2  
   
        result_4=(result_2*result_3)**0.5
        Rho=(result_1/result_4)**2
        print("CCM    libsize=",libsize,", Rho=",Rho,"   ",column_y,"cause",column_x)
        RhoSetCCM.append(Rho)
        libsizeSet.append(libsize)
        return Rho         
   
    Rho=CalculateRho(y_hat,y_tilde_lib_total_sample)  #取样的200个点的相似度
    return Rho


for i_ccm in range(30,lib_total-20,10):    #前面的是x，后面的是y
        CCM(i_ccm,3,4)

plt.plot(libsizeSet,RhoSetCCM,'ro-',label="CCM")
plt.ylim((0, 1))
plt.legend(loc='lower right')
plt.show()
print(libsizeSet)
print(RhoSetCCM)
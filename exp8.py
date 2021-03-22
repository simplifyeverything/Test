'''
Author: your name
Date: 2020-11-18 15:23:29
LastEditTime: 2020-11-24 18:57:21
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \Python_source\DIP\exp8\exp8.py
'''
import cv2 as cv
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
'''
description: 对图像的行和列如果不能被8整除就补0行或0列
param {*} I:输入的图像
return {*}：返回填充后的图像
'''
def fill(I):
    row=I.shape[0]
    column=I.shape[1]
    if row%8!=0:
        I=np.append(np.zeros([int((8-row%8)/2),column]),I,axis=0)#补0
        I=np.append(I,np.zeros([(8-row%8)-int((8-row%8)/2),column]),axis=0)
    row=I.shape[0]
    if column%8!=0:
        I=np.append(np.zeros([row,int((8-column%8)/2)]),I,axis=1)#补0
        I=np.append(I,np.zeros([row,(8-column%8)-int((8-column%8)/2)]),axis=1)
    return I

'''
description:
param {*} I：
return {*}
'''
'''
description: 对图像的每个8x8的小块进行离散余弦变换并量化
param {*} I：输入的图像矩阵
param {*} S：为亮度量化表
return {*}
'''
def Quantization_img(I,S):
    row=I.shape[0]
    column=I.shape[1]
    FDCT_I=np.zeros([row,column])#初始化L量化的
    for i in range(int(row/8)):
        for j in range(int(column/8)):
            chunk=I[8*i:8*(i+1),8*j:8*(j+1)]
            DCT=cv.dct(np.double(chunk))
            FDCT_I[8*i:8*(i+1),8*j:8*(j+1)]=np.round(DCT/S)
    return FDCT_I

'''
description: 对图像的每个8x8的小块进行逆量化并进行逆离散余弦变化
param {*} I：输入的图像矩阵
param {*} S：为亮度量化表
return {*}
'''
def Iquantization_img(I,S):
    row=I.shape[0]
    column=I.shape[1]
    IFDCT_I=np.zeros([row,column])#初始化逆量化的矩阵
    for i in range(int(row/8)):
        for j in range(int(column/8)):
            chunk=I[8*i:8*(i+1),8*j:8*(j+1)]
            Iquantization=S*chunk
            IFDCT_I[8*i:8*(i+1),8*j:8*(j+1)]=cv.idct(np.double(Iquantization))
    return IFDCT_I


s=np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],
            [14,13,16,224,40,57,69,56],[14,17,22,29,51,87,80,62],
            [18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],
            [49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]])#亮度量化表
I=cv.imread("DIP\exp8\Fig0222(a)(face).tif")
I=cv.cvtColor(I,cv.COLOR_RGB2GRAY)
plt.figure()
plt.gray()
plt.subplot(2,2,1)
plt.imshow(I)
plt.title("原图")
plt.axis("off")
plt.subplot(2,2,2)
I=fill(I)
I_Quantized=Quantization_img(I,s)#量化
I_Quantizeds=np.log(np.abs(I_Quantized)+1)
plt.imshow(I_Quantizeds)
plt.title("量化后")
plt.axis("off")
I_IQuantized=Iquantization_img(I_Quantized,s)#解码
plt.subplot(2,2,3)
plt.imshow(I_IQuantized)
plt.title("解码")
plt.axis("off")
plt.show()
PSNR=10*np.log(255**2/(1/64*sum(sum((I_IQuantized-I)*(I_IQuantized-I)))))
print('PSNR=',PSNR)
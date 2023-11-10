import cv2
import numpy as np
import math

# 抽取绿色
def extractGreen(image, kernel_size=3):  # kernel_size 滤波卷积核大小
    # 绿色范围
    lower_green = np.array([35, 43, 46], dtype="uint8")  # 颜色下限[色调，饱和度，亮度]
    upper_green = np.array([77, 255, 255], dtype="uint8")  # 颜色上限
    
    # 高斯滤波
    # img_blur = cv2.GaussianBlur(image, (11, 11), 0)
    # 中值滤波
    img_blur = cv2.medianBlur(image, ksize=kernel_size)
    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    # 根据阈值找到对应颜色，二值化
    mask = cv2.inRange(img_hsv, lower_green, upper_green)
    # 按位与
    output = cv2.bitwise_and(image, image, mask=mask)

    return output

# 图像均衡化
def equalize(image):
    # 分割B,G,R （cv2读取图像的格式即为[B,G,R]，与matplotlib的[R,G,B]不同）
    b,g,r = cv2.split(image)
    b = cv2.equalizeHist(b)
    g = cv2.equalizeHist(g)
    r = cv2.equalizeHist(r)
    equ_img = cv2.merge((b,g,r))
    return equ_img

''' 
以下为ACE实现
'''
 
def stretchImage(data, s=0.005, bins = 2000):    #线性拉伸，去掉最大最小0.5%的像素值，然后线性拉伸至[0,1]
    ht = np.histogram(data, bins)
    d = np.cumsum(ht[0])/float(data.size)
    lmin = 0; lmax=bins-1
    while lmin<bins:
        if d[lmin]>=s:
            break
        lmin+=1
    while lmax>=0:
        if d[lmax]<=1-s:
            break
        lmax-=1
    return np.clip((data-ht[1][lmin])/(ht[1][lmax]-ht[1][lmin]), 0,1)
 
def getPara(radius = 5): #根据半径计算权重参数矩阵
    g_para = {}
    m = g_para.get(radius, None)
    if m is not None:
        return m
    size = radius*2+1
    m = np.zeros((size, size))
    for h in range(-radius, radius+1):
        for w in range(-radius, radius+1):
            if h==0 and w==0:
                continue
            m[radius+h, radius+w] = 1.0/math.sqrt(h**2+w**2)
    m /= m.sum()
    g_para[radius] = m
    return m
 
def zmIce(I, ratio=4, radius=300): #ACE
    para = getPara(radius)
    height,width = I.shape
    zh,zw = [0]*radius + [x for x in range(height)] + [height-1]*radius, [0]*radius + [x for x in range(width)]  + [width -1]*radius
    Z = I[np.ix_(zh, zw)]
    res = np.zeros(I.shape)
    for h in range(radius*2+1):
        for w in range(radius*2+1):
            if para[h][w] == 0:
                continue
            res += (para[h][w] * np.clip((I-Z[h:h+height, w:w+width])*ratio, -1, 1))
    return res
 
def zmIceFast(I, ratio, radius): #单通道快速ACE
    height, width = I.shape[:2]
    if min(height, width) <=2:
        return np.zeros(I.shape)+0.5
    Rs = cv2.resize(I, ((width+1)//2, (height+1)//2))
    Rf = zmIceFast(Rs, ratio, radius)             
    Rf = cv2.resize(Rf, (width, height))
    Rs = cv2.resize(Rs, (width, height))
 
    return Rf+zmIce(I,ratio, radius)-zmIce(Rs,ratio,radius)    
            
def zmIceColor(I, ratio=4, radius=3): #ratio对比度增强因子，radius卷积模板半径
    res = np.zeros(I.shape)
    for k in range(3):
        res[:,:,k] = stretchImage(zmIceFast(I[:,:,k], ratio, radius))
    return res

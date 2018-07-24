import numpy as np
import cv2

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def get_bird_point(cnts):
    max_num = 0
    res = (0, 0)
    for cnt in cnts:
        x = 0
        y = 0
        num = 0
        max_x = 0
        for point in cnt:
            x += point[0][0]
            y += point[0][1]
            num += 1
            if max_x < point[0][0]:
                max_x = point[0][0]
        if max_num < num and max_x < 120:
            max_num = num
            res = (int(x/num), int(y/num))
    if((res[0]==0 and res[1]==0) or max_num < 20):
        res = (100,350)
    return res


def cv_img2mat_img(img):
    img2 = np.zeros(np.shape(img),dtype='uint8')
    img2[:,:,0] = img[:,:,2]
    img2[:,:,1] = img[:,:,1]
    img2[:,:,2] = img[:,:,0]
    return img2

# 求灰度图像众数, uint8
def gray_img_data_mode(img_gray):
    w, h = np.shape(img_gray)
    data = np.reshape(img_gray, w*h)
    return np.argmax(np.bincount(data))

def color_img_data_mode(img):
    modes = []
    for c in range(3):
        img_c = img[:,:,c]
        w, h = np.shape(img_c)
        data = np.reshape(img_c, w*h)
        modes.append(np.argmax(np.bincount(data)))
    return modes

def get_feature_points(img):
    img = img[0:400, :, :]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mode = gray_img_data_mode(img_gray)
    img_gray[img_gray[:,:] == mode] = 0


    img_bit = np.zeros(img_gray.shape).astype('uint8')
    img_bit[img_gray[:, :] > 0] = 1
    
    # 获取连通域
    image, cnts, hierarchy = cv2.findContours(img_bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bird_point = get_bird_point(cnts)
    bird_x = bird_point[0]
    row_sum = np.sum(img_bit, 0)
    obs_x = np.argmax(row_sum[bird_x:] > 200) + bird_x
    col_val = img_bit[:,obs_x]
    obs_y1 = np.argmax(col_val==0)
    obs_y2 = len(col_val) - np.argmax(col_val[::-1]==0)
    if(obs_y1==0 or obs_y2==400):
        obs_x = 280
        obs_y1=100
        obs_y2=300
    return np.array([bird_point[0],bird_point[1],obs_x,obs_y1,obs_x,obs_y2])

def get_feature_distances(img):
    b_x, b_y, o_x1, o_y1, o_x2, o_y2 = get_feature_points(img)
    return np.array([o_x1-b_x,o_y1-b_y,o_y2-b_y])
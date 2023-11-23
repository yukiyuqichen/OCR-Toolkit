import load_files
import cv2
import numpy as np


# 根据联通区域或轮廓的面积去除孤立点
def delete_isolated_points(img):
    # 设定要去除的孤立点最大面积
    threshold = 50

    # 二值化
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    except Exception as e:
        gray = img  # 处理文件本身已经为灰值图的情况
    # 图像取非（将黑白像素颠倒，若不进行此操作，则无法检测到黑色文本像素的连通区域）
    grayNot = cv2.bitwise_not(gray)
    # 取非后的二值图
    ret, img_binary = cv2.threshold(grayNot, 175, 255, cv2.THRESH_BINARY)  # 阈值硬编码为175，之后可以改

    # 算法1：检测连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_binary, connectivity=4, ltype=None)
    ## 背景为黑色(0)
    # new_img = np.zeros((img_binary.shape[0], img_binary.shape[1]), np.uint8)
    # 背景为白色(255)
    new_img = np.ones((img_binary.shape[0], img_binary.shape[1]), np.uint8)
    new_img = new_img * 255

    ## 给不同的连通区域上色后返回图像
    # colors = []
    # for i in range(num_labels):
    #     b = np.random.randint(0, 256)
    #     g = np.random.randint(0, 256)
    #     r = np.random.randint(0, 256)
    #     colors.append((b, g, r))
    # colors[0] = (0, 0, 0)
    # h, w = gray.shape
    # image = np.zeros((h, w, 3), dtype=np.uint8)
    # for row in range(h):
    #     for col in range(w):
    #         image[row, col] = colors[labels[row, col]]
    # return image

    for i in range(1, num_labels):
        mask = labels == i
        if stats[i][4] > threshold:
            new_img[mask] = 0
        else:
            new_img[mask] = 255
    return new_img

    # # 算法2：检测轮廓
    # contours, hierarch = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(len(contours))
    # for i in range(len(contours)):
    #     area = cv2.contourArea(contours[i])
    #     if area < threshold:
    #         cv2.drawContours(img_binary, [contours[i]], -1, (255, 255, 255), thickness=-1)
    #         continue
    # return img_binary



def pre_denoise(pre_select_dir_path, pre_save_dir_path, frame_pre, process, file_num, count_num):
    process.set("Start!")
    frame_pre.update()
    # 读取图片文件
    filelist = load_files.load_img(pre_select_dir_path)
    file_len = len(filelist)
    file_num.set(file_len)
    count = 0
    count_num.set(0)
    file_path = pre_select_dir_path.get() + '/'
    save_path = pre_save_dir_path.get() + '/'

    for file in filelist:
        img = cv2.imdecode(np.fromfile(file_path+file, dtype=np.uint8), -1)  # 可读取中文文件名
        img_denoised = delete_isolated_points(img)
        # 存储校正后的图片
        cv2.imencode('.jpg', img_denoised)[1].tofile(save_path + str(file).replace(".*", "") + "_corrected" + ".png")  # 可存储中文文件名
        # 打印进度
        count += 1
        count_num.set(count)
        process.set("Processing: " + str(count_num.get()) + " / " + str(file_num.get()))
        frame_pre.update()
        print("Finish: " + str(count) + " / " + str(file_len))
    process.set("Finished!")
    frame_pre.update()
    print("Finished!")

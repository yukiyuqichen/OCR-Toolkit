import cv2
from PIL import Image
import numpy as np
import load_files


# 垂直投影
def vertical_project(binary, h, w):
    projection = np.sum(binary, axis=0) / 255
    return projection


# 根据垂直投影找出分界线
def split_without_line(projection, h, w):
    score = h   # 在该垂直方向上所有的像素均为白色，则满足score=h，可作为分界线的备选位置
    projection = np.array(projection)   # 将列表转换为矩阵
    projection_binary = np.where(projection > score-1, 255, 0)  # 将score作为阈值，将矩阵二值化
    projection_img = np.ones((100, w), np.uint8)    # 将矩阵转换为图像
    projection_img = np.multiply(projection_img, projection_binary)    # 将矩阵转换为图像
    projection_img = projection_img.astype('uint8')     # 重要：将格式设为unit8

    # 膨胀后求连通区域
    # kernel = np.ones((3, 3), dtype=np.uint8)
    # dilate = cv2.dilate(projection_img, kernel, 1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(projection_img, connectivity=8, ltype=None)

    # 求中心点横坐标最靠近图像中心的值
    middle = w/2
    centroids_x = []
    centroids_x_minus_middle = []
    for i in range(num_labels):
        centroids_x.append(centroids[i][0])
    for i in range(len(centroids_x)):
        centroids_x_minus_middle.append(abs(centroids_x[i] - middle))
    x = centroids_x[centroids_x_minus_middle.index(min(centroids_x_minus_middle))]
    return x

    # # 求面积排前三的连通区域及其中心点的横坐标
    # stats_top_3 = []
    # labels_top_3 = []
    # centroids_x_top_3 = []
    # for i in range(3):
    #     stats_top_3.append(stats[i][-1])
    #     labels_top_3.append(i)
    # if num_labels > 3:
    #     for i in range(3, num_labels):
    #         if stats[i][-1] > min(stats_top_3):
    #             pop_index = stats_top_3.index(min(stats_top_3))
    #             del stats_top_3[pop_index]
    #             del labels_top_3[pop_index]
    #             stats_top_3.append(stats[i][-1])
    #             labels_top_3.append(i)
    # for i in labels_top_3:
    #     centroids_x_top_3.append(centroids[i][0])
    #     print(centroids[i][0])
    #     print(centroids[i][1])
    # # 为中心点横坐标列表排序，并选择排中间的x坐标
    # centroids_x_top_3.sort()
    # x = centroids_x_top_3[1]
    # return x


def pre_split_without_line(pre_select_dir_path, pre_save_dir_path, frame_pre, process, file_num, count_num):
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
        img_data = cv2.imdecode(np.fromfile(file_path + file, dtype=np.uint8), -1)  # 可读取中文文件名
        try:
            gray = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
        except Exception as e:
            gray = img_data     # 处理文件本身已经为灰值图的情况
        # 固定阈值二值化
        ret, img_binary = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)      # 这里的阈值很重要，阈值过小则可能导致切分错误
        # 自适应阈值二值化
        # img_binary = cv2.adaptiveThreshold(grayNot, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

        # 根据垂直投影找出分界线
        h, w = img_binary.shape
        projection = vertical_project(img_binary, h, w)
        x = split_without_line(projection, h, w)
        point_top_x = x
        point_down_x = x

        # 通过遮挡间接实现图像切分，分别存储左右两张图像
        left_img_points = np.array([[(point_top_x, 0), (w, 0), (w, h), (point_down_x, h)]], dtype=np.int32)
        right_img_points = np.array([[(0, 0), (point_top_x, 0), (point_down_x, h), (0, h)]], dtype=np.int32)
        left_img = np.copy(img_data)
        right_img = np.copy(img_data)
        cv2.fillConvexPoly(left_img, left_img_points, (255, 255, 255))
        # cv2.imwrite(save_path + str(file).replace(".*", "") + "_left" + ".png", left_img)
        cv2.imencode('.jpg', left_img)[1].tofile(save_path + str(file).replace(".*", "") + "_left" + ".png")  # 可存储中文文件名
        cv2.fillConvexPoly(right_img, right_img_points, (255, 255, 255))
        # cv2.imwrite(save_path + str(file).replace(".*", "") + "_right" + ".png", right_img)
        cv2.imencode('.jpg', right_img)[1].tofile(
            save_path + str(file).replace(".*", "") + "_right" + ".png")  # 可存储中文文件名
        # 打印进度
        count += 1
        count_num.set(count)
        process.set("Processing: " + str(count_num.get()) + " / " + str(file_num.get()))
        frame_pre.update()
        print("Finish: " + str(count) + " / " + str(file_len))
    process.set("Finished!")
    frame_pre.update()
    print("Finished!")

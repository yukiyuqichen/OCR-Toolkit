import os
import cv2
import numpy as np
from sklearn.cluster import KMeans


# # 单元格之间无界划，只有空白，可用垂直或水平投影求分界线
# def project(img):
#
#
# # 单元格之间有界划，可用霍夫变换求分界线
# def hough_lines():
#
#
# # 单元格之间有界划，也可用膨胀腐蚀法求界线
# def erode_dilate():
#

# 表格版面分析
def layout_table(file_path, save_path):
    files = os.listdir(file_path)
    count = 0
    for file in files:
        # 图像预处理
        img = cv2.imdecode(np.fromfile(file_path+file, dtype=np.uint8), -1)  # 可读取中文文件名

        # 获取灰度图
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            gray = img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 灰度图取反
        grayNot = cv2.bitwise_not(gray)

        ## 固定阈值二值化
        ret, binary = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
        # 自适应阈值二值化
        # binary = cv2.adaptiveThreshold(grayNot, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        #cv2.imwrite(save_path+str(file).replace(".*", "")+"_binary"+".png", binary)   # 存储二值化图

        # 获取轮廓
        edges = cv2.Canny(binary, 50, 125, apertureSize=3)
        # cv2.imwrite(save_path+str(file).replace(".*", "")+"_edges"+".png", edges)     # 存储轮廓图


        # 霍夫变换检测直线
        lines = cv2.HoughLinesP(edges, 1.0, np.pi/180, 100, np.array([]), minLineLength=100, maxLineGap=5)

        # 去除横线，仅保留竖线
        vertical_lines = []
        for i in range(len(lines)):
            if -80 < lines[i][0][2] - lines[i][0][0] < 90:
                vertical_lines.append(lines[i])
        lines = vertical_lines

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #cv2.imwrite(save_path+str(file).replace(".*", "")+"_lines"+".png", img)    # 存储标注直线后的图像


        height = edges.shape[0]
        width = edges.shape[1]
        h = height - 1
        w = width - 1

        # 将检测到的直线线段保存为以起始两点横坐标元组为键，以起始两点纵坐标元组为值的字典
        lines_dict = {}
        for line in lines:
            x_tuple = tuple(line[0][0::2])
            y_tuple = tuple(line[0][1::2])
            lines_dict[x_tuple] = y_tuple

        # 通过lambda对lines列表中的每个线段进行切片，将每个线段起始两点的横坐标存入矩阵，用于Kmeans聚类
        x_list = list(map(lambda l: l[0][0::2], lines))
        y_list = list(map(lambda l: l[0][1::2], lines))
        x_array = np.array(x_list)

        # 情况0 没有检测到任何直线，len(x_list) == 0; 目前暂未考虑进这种情况，且由于在for循环中本次未被重新赋值的值会引用上次的值，因此即使出现这种情况，程序也不会报错；需要在霍夫转换的调参环节进行测试，规避无法检测到任何直线的情况
        # 情况1 检测到的直线线段仅有一个，无需使用KMeans
        # print(x_list)
        if len(x_list) == 1:
            point_x_1 = x_list[0][0]
            point_x_2 = x_list[0][1]
            point_y_1 = y_list[0][0]
            point_y_2 = y_list[0][1]

        # 情况2 检测到的直线线段大于一个，通过KMeans聚类对所有线段进行聚类，输入数据是线段起始两点的横坐标
        if len(x_list) > 1:
            kmeans = KMeans(n_clusters=5)
            kmeans.fit(x_array)
            cluster_assignment = kmeans.predict(x_array)
            cluster_list = list(cluster_assignment)
            # 对每一簇对应线段所有x坐标和y坐标分别求方差
            x_list_0 = []
            y_list_0 = []
            x_list_1 = []
            y_list_1 = []
            x_list_2 = []
            y_list_2 = []
            x_list_3 = []
            y_list_3 = []
            x_list_4 = []
            y_list_4 = []
            for i in range(len(list(cluster_list))):
                if cluster_list[i] == 0:
                    x_list_0.append(x_list[i][0])
                    x_list_0.append(x_list[i][1])
                    y_list_0.append(y_list[i][0])
                    y_list_0.append(y_list[i][1])
                if cluster_list[i] == 1:
                    x_list_1.append(x_list[i][0])
                    x_list_1.append(x_list[i][1])
                    y_list_1.append(y_list[i][0])
                    y_list_1.append(y_list[i][1])
                if cluster_list[i] == 2:
                    x_list_2.append(x_list[i][0])
                    x_list_2.append(x_list[i][1])
                    y_list_2.append(y_list[i][0])
                    y_list_2.append(y_list[i][1])
                if cluster_list[i] == 3:
                    x_list_3.append(x_list[i][0])
                    x_list_3.append(x_list[i][1])
                    y_list_3.append(y_list[i][0])
                    y_list_3.append(y_list[i][1])
                if cluster_list[i] == 4:
                    x_list_4.append(x_list[i][0])
                    x_list_4.append(x_list[i][1])
                    y_list_4.append(y_list[i][0])
                    y_list_4.append(y_list[i][1])

            x_list_list = []
            y_list_list = []
            x_list_list.append(x_list_0)
            x_list_list.append(x_list_1)
            x_list_list.append(x_list_2)
            x_list_list.append(x_list_3)
            x_list_list.append(x_list_4)
            y_list_list.append(y_list_0)
            y_list_list.append(y_list_1)
            y_list_list.append(y_list_2)
            y_list_list.append(y_list_3)
            y_list_list.append(y_list_4)

            new_img = binary

            order = []
            for i in range(len(x_list_list)):
                x_order = x_list_list[i][0]
                order.append(x_order)

            order_2 = order.copy()

            for i in range(len(x_list_list)):
                index = order.index(min(order_2))
                index_2 = order_2.index(min(order_2))
                print(order)
                print(order_2)
                print(min(order_2))
                print(index)
                order_2.pop(index_2)
                point_x_1 = x_list_list[index][0]
                point_x_2 = x_list_list[index][1]
                point_y_1 = y_list_list[index][0]
                point_y_2 = y_list_list[index][1]


                # 算出中间边界栏（y=kx+b）在图片顶端与底端的交点的坐标
                point_top_y = 0
                point_down_y = h
                # (point_top_y - point_y_1) / (point_top_x - point_x_1) = (point_y_2 - point_y_1) / (point_x_2 - point_x_1)
                point_top_x = ((point_top_y - point_y_1) * (point_x_2 - point_x_1) / (point_y_2 - point_y_1)) + point_x_1
                point_down_x = ((point_down_y - point_y_1) * (point_x_2 - point_x_1) / (point_y_2 - point_y_1)) + point_x_1
                # # 为了避免浮点数溢出，而使用函数对浮点数进行截断
                # point_top_x = np.round((np.round(((point_top_y - point_y_1) * (point_x_2 - point_x_1)), 5) / np.round((point_y_2 - point_y_1), 5) + point_x_1), 5)
                # point_down_x = np.round((np.round(((point_down_y - point_y_1) * (point_x_2 - point_x_1)), 5) / np.round((point_y_2 - point_y_1), 5) + point_x_1), 5)
                # print(point_x_1, point_x_2, point_y_1, point_y_2)
                # print(point_down_x)
                # print(point_top_x)

                # 通过遮挡间接实现图像切分，分别存储左右两张图像
                left_img_points = np.array([[(point_top_x, 0), (w, 0), (w, h), (point_down_x, h)]], dtype=np.int32)
                right_img_points = np.array([[(0, 0), (point_top_x, 0), (point_down_x, h), (0, h)]], dtype=np.int32)
                left_img = np.copy(new_img)
                right_img = np.copy(new_img)

                cv2.fillConvexPoly(left_img, left_img_points, (255, 255, 255))
                cv2.imwrite(save_path+str(file).replace(".*", "") + "_" + str(i) + "_left"+".png", left_img)

                cv2.fillConvexPoly(right_img, right_img_points, (255, 255, 255))
                cv2.imwrite(save_path+str(file).replace(".*", "") + "_" + str(i) + "_right"+".png", right_img)

                cv2.imwrite(save_path + str(file).replace(".*", "") + "_" + str(i) + "_new" + ".png", new_img)
                new_img = right_img



                # 打印进度条
                count += 1
                print("Finish: " + str(count) + " / " + str(len(files)))


def main():
    file_path = "C:/Users/Moon/Desktop/TEST/"
    save_path = "C:/Users/Moon/Desktop/RESULT/"
    layout_table(file_path, save_path)


if __name__ == '__main__':
    main()

import load_files
import numpy as np
import cv2
from sklearn.cluster import KMeans


# 根据中间界栏分割页面
def pre_split(pre_select_dir_path, pre_save_dir_path, frame_pre, process, file_num, count_num):
    process.set("")
    frame_pre.update()
    # 读取图片文件
    filelist = load_files.load_img(pre_select_dir_path)
    file_len = len(filelist)
    file_num.set(file_len)
    count = 0
    count_num.set(0)
    # 对图片文件进行二值化
    file_path = pre_select_dir_path.get() + '/'
    save_path = pre_save_dir_path.get() + '/'
    for file in filelist:
        # 图像预处理
        img = cv2.imdecode(np.fromfile(file_path+file, dtype=np.uint8), -1)  # 可读取中文文件名
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            gray = img
        ret, binary = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(binary, 50, 125, apertureSize=3)
        height = edges.shape[0]
        width = edges.shape[1]
        h = height - 1
        w = width - 1
        # w_1 = int((width - 1) / 2 - 150)
        # w_2 = int((width - 1) / 2 + 150)
        w_1 = 0
        w_2 = 200

        # 设置ROI与遮罩
        ROI = np.array([[(w_1, height), (w_2, height), (w_2, 0), (w_1, 0)]])
        black_img = np.zeros_like(edges)
        mask = cv2.fillPoly(black_img, ROI, 255)
        masked_img = cv2.bitwise_and(edges, mask)

        # 霍夫变换检测直线
        lines = cv2.HoughLinesP(masked_img, 1.0, np.pi / 180, 100, np.array([]), minLineLength=110, maxLineGap=10)
        # 去除横线，仅保留竖线
        vertical_lines = []
        for i in range(len(lines)):
            if -80 < lines[i][0][2] - lines[i][0][0] < 90:
                vertical_lines.append(lines[i])
        lines = vertical_lines

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

        # 情况1 检测到的直线线段仅有一个，无需使用KMeans
        print(x_list)
        if len(x_list) == 1:
            point_x_1 = x_list[0][0]
            point_x_2 = x_list[0][1]
            point_y_1 = y_list[0][0]
            point_y_2 = y_list[0][1]

        # 情况2 检测到的直线线段大于一个，通过KMeans聚类对所有线段进行聚类，输入数据是线段起始两点的横坐标
        if len(x_list) > 1:
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(x_array)
            cluster_assignment = kmeans.predict(x_array)
            cluster_list = list(cluster_assignment)
            # 对每一簇对应线段所有x坐标和y坐标分别求方差
            x_list_0 = []
            y_list_0 = []
            x_list_1 = []
            y_list_1 = []
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

            try:
                x_0_var = np.var(x_list_0)
                y_0_var = np.var(y_list_0)
                x_1_var = np.var(x_list_1)
                y_1_var = np.var(y_list_1)
            except Exception:
                print("There are some errors about variance!")
            if x_0_var == 0:
                x_0_var = 1
            if y_0_var == 0:
                y_0_var = 1
            if x_1_var == 0:
                x_1_var = 1
            if y_1_var == 0:
                y_1_var = 1
            # 根据方差信息找到归属于中间边界栏的竖线线段的聚类，排除其它干扰
            # 归属于中间边界栏的线段的x坐标分布密集，方差小，y坐标分布稀疏大，因此y坐标/x坐标之比大
            # 从归属于中间边界栏的线段中任意选择一个（此处暂时选择第一个），用来计算y=kx+b在图片顶端与底端的交点
            ratio_0 = y_0_var / x_0_var
            ratio_1 = y_1_var / x_1_var
            if ratio_0 > ratio_1:
                point_x_1 = x_list_0[0]
                point_x_2 = x_list_0[1]
                point_y_1 = y_list_0[0]
                point_y_2 = y_list_0[1]
            else:
                if ratio_0 < ratio_1:
                    point_x_1 = x_list_1[0]
                    point_x_2 = x_list_1[1]
                    point_y_1 = y_list_1[0]
                    point_y_2 = y_list_1[1]
                else:
                    print("There are some errors about the ratio!")

        # 算出中间边界栏（y=kx+b）在图片顶端与底端的交点的坐标
        point_top_y = 0
        point_down_y = h
        # (point_top_y - point_y_1) / (point_top_x - point_x_1) = (point_y_2 - point_y_1) / (point_x_2 - point_x_1)
        point_top_x = ((point_top_y - point_y_1) * (point_x_2 - point_x_1) / (point_y_2 - point_y_1)) + point_x_1
        point_down_x = ((point_down_y - point_y_1) * (point_x_2 - point_x_1) / (point_y_2 - point_y_1)) + point_x_1
        # # 为了避免浮点数溢出，而使用函数对浮点数进行截断
        # point_top_x = np.round((np.round(((point_top_y - point_y_1) * (point_x_2 - point_x_1)), 5) / np.round((point_y_2 - point_y_1), 5) + point_x_1), 5)
        # point_down_x = np.round((np.round(((point_down_y - point_y_1) * (point_x_2 - point_x_1)), 5) / np.round((point_y_2 - point_y_1), 5) + point_x_1), 5)
        print(point_x_1, point_x_2, point_y_1, point_y_2)
        print(point_down_x)
        print(point_top_x)
        # 通过遮挡间接实现图像切分，分别存储左右两张图像
        left_img_points = np.array([[(point_top_x, 0), (w, 0), (w, h), (point_down_x, h)]], dtype=np.int32)
        right_img_points = np.array([[(0, 0), (point_top_x, 0), (point_down_x, h), (0, h)]], dtype=np.int32)
        left_img = np.copy(binary)
        right_img = np.copy(binary)
        cv2.fillConvexPoly(left_img, left_img_points, (255, 255, 255))
        # cv2.imwrite(save_path + str(file).replace(".*", "") + "_left" + ".png", left_img)
        cv2.imencode('.jpg', left_img)[1].tofile(save_path + str(file).replace(".*", "") + "_left" + ".png")  # 可存储中文文件名
        cv2.fillConvexPoly(right_img, right_img_points, (255, 255, 255))
        # cv2.imwrite(save_path + str(file).replace(".*", "") + "_right" + ".png", right_img)
        cv2.imencode('.jpg', right_img)[1].tofile(save_path + str(file).replace(".*", "") + "_right" + ".png")  # 可存储中文文件名
        # 打印进度
        count += 1
        count_num.set(count)
        process.set("Processing: " + str(count_num.get()) + " / " + str(file_num.get()))
        frame_pre.update()
        print("Finish: " + str(count) + " / " + str(file_len))
    process.set("Finished!")
    frame_pre.update()
    print("Finished!")

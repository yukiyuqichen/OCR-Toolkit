import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import os
import numpy as np
import imghdr
import cv2
import base64
import requests
import urllib
from sklearn.cluster import KMeans


# 用户选择路径
# tab_pre中的选择路径
def pre_select_dir():
    selected_dir_path = filedialog.askdirectory()
    pre_select_dir_path.set(selected_dir_path)


def pre_save_dir():
    selected_dir_path = filedialog.askdirectory()
    pre_save_dir_path.set(selected_dir_path)


# tab_ocr中的选择路径
def ocr_select_dir():
    selected_dir_path = filedialog.askdirectory()
    ocr_select_dir_path.set(selected_dir_path)


def ocr_save_dir():
    selected_dir_path = filedialog.askdirectory()
    ocr_save_dir_path.set(selected_dir_path)


# 图片二值化
def pre_binary():
    process.set("")
    process_end.set("")
    frame_pre.update()
    filelist = os.listdir(pre_select_dir_path.get())
    # 略过非图片文件，将图片文件存入新的列表
    new_filelist = []
    for file in filelist:
        file_path = pre_select_dir_path.get() + '/' + file
        img_type = {'jpg', 'jpeg', 'bmp', 'png', 'tif', 'tiff'}
        if imghdr.what(file_path) not in img_type:
            continue
        new_filelist.append(file)
    # 对新列表中的图片文件进行二值化
    file_len = len(new_filelist)
    file_num.set(file_len)
    count = 0
    count_num.set(0)
    for file in new_filelist:
        file_path = pre_select_dir_path.get() + '/' + file
        new_file_path = pre_save_dir_path.get() + '/' + file
        img_data = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8),-1)  # 可读取中文文件名
        try:
            gray = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
        except Exception as e:
            gray = img_data
        ret, img_binary = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
        cv2.imencode('.jpg', img_binary)[1].tofile(new_file_path)   # 可存储中文文件名
        # 打印进度
        count += 1
        count_num.set(count)
        process.set("Processing: " + str(count_num.get()) + " / " + str(file_num.get()))
        frame_pre.update()
        print("Finish: " + str(count) + " / " + str(file_len))
    process_end.set("Finished!")
    frame_pre.update()
    print("Finished!")


# 根据中间界栏分割页面
def pre_split():
    process.set("")
    process_end.set("")
    frame_pre.update()
    filelist = os.listdir(pre_select_dir_path.get())
    # 略过非图片文件，将图片文件存入新的列表
    new_filelist = []
    for file in filelist:
        file_path = pre_select_dir_path.get() + '/' + file
        img_type = {'jpg', 'jpeg', 'bmp', 'png', 'tif', 'tiff'}
        if imghdr.what(file_path) not in img_type:
            continue
        new_filelist.append(file)
    # 对新列表中的图片文件进行二值化
    file_len = len(new_filelist)
    file_num.set(file_len)
    count = 0
    count_num.set(0)
    file_path = pre_select_dir_path.get() + '/'
    save_path = pre_save_dir_path.get() + '/'
    for file in new_filelist:
        # 图像预处理
        img = cv2.imdecode(np.fromfile(file_path+file, dtype=np.uint8), -1)  # 可读取中文文件名
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            gray = img
        ret, binary = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
        # cv2.imwrite(save_path+str(file).replace(".*", "")+"_binary"+".png", binary)   # 存储二值化图
        edges = cv2.Canny(binary, 50, 125, apertureSize=3)
        # cv2.imwrite(save_path+str(file).replace(".*", "")+"_edges"+".png", edges)     # 存储轮廓图
        height = edges.shape[0]
        width = edges.shape[1]
        h = height - 1
        w = width - 1
        w_1 = int((width - 1) / 2 - 150)
        w_2 = int((width - 1) / 2 + 150)

        # 设置ROI与遮罩
        ROI = np.array([[(w_1, height), (w_2, height), (w_2, 0), (w_1, 0)]])
        black_img = np.zeros_like(edges)
        mask = cv2.fillPoly(black_img, ROI, 255)
        masked_img = cv2.bitwise_and(edges, mask)
        # cv2.imwrite(save_path+str(file).replace(".*", "")+"_mask"+".png", masked_img)    # 存储遮罩

        # 霍夫变换检测直线
        # lines = cv2.HoughLinesP(masked_img, 1.0, np.pi/180, 100, np.array([]), minLineLength=120, maxLineGap=10)  # 在该参数下，1777张测试集中有2张无法检测到任何直线
        lines = cv2.HoughLinesP(masked_img, 1.0, np.pi / 180, 100, np.array([]), minLineLength=110, maxLineGap=10)
        # 去除横线，仅保留竖线
        vertical_lines = []
        for i in range(len(lines)):
            if -80 < lines[i][0][2] - lines[i][0][0] < 90:
                vertical_lines.append(lines[i])
        lines = vertical_lines

        # for line in lines:
        #     x1, y1, x2, y2 = line[0]
        #     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 10)
        # cv2.imwrite(save_path + str(file).replace(".*", "") + "_lines" + ".png", img)  # 存储标注直线后的图像

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
            # print(y_0_var, x_0_var, ratio_0, "", y_1_var, x_1_var, ratio_1)
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
    process_end.set("Finished!")
    frame_pre.update()
    print("Finished!")


# 前处理
def preprocess():
    if pre_option.get() == "Binary (去噪黑白化)":
        pre_binary()
    if pre_option.get() == "Split (根据中间界栏分割图片)":
        pre_split()


# 接入百度OCR
def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": api_key.get(), "client_secret": secret_key.get()}
    return str(requests.post(url, params=params).json().get("access_token"))


def ocr(image):
    with open(image, 'rb') as f:
        image_data = f.read()
        base64_data = base64.b64encode(image_data)  # 对图片进行base64编码
        base64_data_url = urllib.parse.quote(base64_data)  # 对base64编码进行url编码
    url = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic?access_token=" + get_access_token()    # 高精度不含位置版
    payload = 'image=' + base64_data_url + '&language_type=CHN_ENG' + '&probability=true' + '&detect_direction=true'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    response_json = response.json()
    if 'error_msg' not in response_json.keys():
        token.set("Successfully authenticated.")
        frame_ocr.update()
        return response_json
    else:
        token.set("Error: " + response_json['error_msg'])
        frame_ocr.update()


def general_ocr():
    token.set("")
    process.set("")
    process_end.set("")
    frame_pre.update()
    filelist = os.listdir(ocr_select_dir_path.get())
    text_write = []
    probability_write = []
    # 略过非图片文件，将图片文件存入新的列表
    new_filelist = []
    for file in filelist:
        file_path = ocr_select_dir_path.get() + '/' + file
        img_type = {'jpg', 'jpeg', 'bmp', 'png', 'tif', 'tiff'}
        if imghdr.what(file_path) not in img_type:
            continue
        new_filelist.append(file)
    # 对新列表中的图片文件进行ocr
    file_len = len(new_filelist)
    file_num.set(file_len)
    count = 0
    count_num.set(0)
    for file in new_filelist:
        file_path = ocr_select_dir_path.get() + '/' + file
        json = ocr(file_path)
        text = json["words_result"]
        # 存储置信度至txt
        probability_write.append("PROBABILITY FROM: " + str(file) + "\n")
        for i in text:
            probability_write.append("average:" + str(i['probability']['average']) + "," + "min:" + str(i['probability']['min']) + "," + "variance:" + str(i['probability']['variance']))
            probability_write.append("\n")
        probability_save = ocr_save_dir_path.get() + '/' + 'OutputProbability.txt'
        with open(probability_save, 'w', encoding='utf-8-sig', errors='ignore') as f:
            f.writelines(probability_write)
        # 存储文本至txt
        text_write.append("TEXT FROM: " + str(file) + "\n")
        for i in text:
            text_write.append(i['words'])
            text_write.append("\n")
        text_save = ocr_save_dir_path.get() + '/' + 'Output.txt'
        with open(text_save, 'w', encoding='utf-8-sig', errors='ignore') as f:  # 将文件夹内所有图片的对应文本写入同一个txt文件
            f.writelines(text_write)
        # 打印进度
        count += 1
        count_num.set(count)
        process.set("Processing: " + str(count_num.get()) + " / " + str(file_num.get()))
        frame_ocr.update()
        print("Finish: " + str(count) + " / " + str(file_len))
    process_end.set("Finished!")
    frame_ocr.update()
    print("Finished!")


# 创建主窗口
root_window = tk.Tk()
# 设置主窗口标题
root_window.title('OCR Tookit')
# 设置主窗口大小
root_window.geometry('350x300')

# 设置标签卡
tab = ttk.Notebook(root_window)
frame_pre = tk.Frame(tab)
frame_ocr = tk.Frame(tab)
frame_about = tk.Frame(tab)
tab.add(frame_pre, text="Pre Process")
tab.add(frame_ocr, text="OCR")
tab.add(frame_about, text="About")
tab.pack(expand=1, fill='both')

# 设置公共变量
# 进度变量
token = tk.StringVar()
count_num = tk.IntVar()
file_num = tk.IntVar()
process = tk.StringVar()
process_end = tk.StringVar()


# 设置frame_pre中的变量
# 路径变量
pre_select_dir_path = tk.StringVar()    # 准备OCR的图片文件夹所在路径
pre_save_dir_path = tk.StringVar()      # OCR完成后的文本文件存储路径
# 选项变量
pre_option = tk.StringVar()


# 设置frame_pre中的组件
# 作装饰的空label
tk.Label(frame_pre, text="      ").grid(row=0, column=1)
# 下拉菜单
tk.Label(frame_pre, text="Option: ").grid(row=1, column=2)
com = ttk.Combobox(frame_pre, textvariable=pre_option)
com["value"] = ("Binary (去噪黑白化)", "Split (根据中间界栏分割图片)")
com.current(0)
com.grid(row=1, column=3)
# 路径选择组件
tk.Button(frame_pre, text="Image Path", command=pre_select_dir).grid(row=2, column=2)
tk.Entry(frame_pre, textvariable=pre_select_dir_path).grid(row=2, column=3)
tk.Button(frame_pre, text="Output Path", command=pre_save_dir).grid(row=3, column=2)
tk.Entry(frame_pre, textvariable=pre_save_dir_path).grid(row=3, column=3)
# 开始运行按钮
tk.Button(frame_pre, text="Start", command=preprocess).grid(row=4, column=4)
# 作装饰的空label
tk.Label(frame_pre, text="").grid(row=5, column=1)
# 进度条显示
tk.Label(frame_pre, textvariable=process).grid(row=6, column=1, columnspan=3)
tk.Label(frame_pre, textvariable=process_end).grid(row=7, column=1, columnspan=3)


# 设置frame_ocr中的变量
# 路径变量
ocr_select_dir_path = tk.StringVar()    # 准备OCR的图片文件夹所在路径
ocr_save_dir_path = tk.StringVar()      # OCR完成后的文本文件存储路径
# Token变量
api_key = tk.StringVar()
secret_key = tk.StringVar()


# 设置frame_ocr中的组件
# 作装饰的空label
tk.Label(frame_ocr, text="      ").grid(row=0, column=1)
# api_key输入组件
tk.Label(frame_ocr, text="API_KEY: ").grid(row=1, column=2)
tk.Entry(frame_ocr, textvariable=api_key).grid(row=1, column=3)
tk.Label(frame_ocr, text="SECRET_KEY: ").grid(row=2, column=2)
tk.Entry(frame_ocr, textvariable=secret_key).grid(row=2, column=3)
# 路径选择组件
tk.Button(frame_ocr, text="Image Path", command=ocr_select_dir).grid(row=3, column=2)
tk.Entry(frame_ocr, textvariable=ocr_select_dir_path).grid(row=3, column=3)
tk.Button(frame_ocr, text="Output Path", command=ocr_save_dir).grid(row=4, column=2)
tk.Entry(frame_ocr, textvariable=ocr_save_dir_path).grid(row=4, column=3)
# 开始运行按钮
tk.Button(frame_ocr, text="Start", command=general_ocr).grid(row=6, column=4)
# 作装饰的空label
tk.Label(frame_ocr, text="").grid(row=7, column=1)
# 进度条显示
tk.Label(frame_ocr, textvariable=token).grid(row=8, column=1, columnspan=3)
tk.Label(frame_ocr, textvariable=process).grid(row=9, column=1, columnspan=3)
tk.Label(frame_ocr, textvariable=process_end).grid(row=10, column=1, columnspan=3)


# 设置frame_about中的组件
tk.Label(frame_about, text="                    ").grid(row=0, column=0)
tk.Label(frame_about, text="For CBDB group").grid(row=1, column=1)
tk.Label(frame_about, text="Developed by: Yuqi Chen").grid(row=2, column=1)
tk.Label(frame_about, text="Email: cyq0722@pku.edu.cn").grid(row=3, column=1)
tk.Label(frame_about, text="Thanks to: Hongsu Wang").grid(row=4, column=1)
tk.Label(frame_about, text="                    ").grid(row=5, column=0)
tk.Label(frame_about, text="Icon drawn by: DALL·E 2").grid(row=6, column=1)
tk.Label(frame_about, text="Version: 2023.02.28").grid(row=7, column=1)

# 进入主循环，显示主窗口
root_window.mainloop()



import numpy as np
import cv2
# 图片二值化
import load_files


def pre_binary(pre_select_dir_path, pre_save_dir_path, frame_pre, process, file_num, count_num):
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
    # 对图片文件进行二值化
    for file in filelist:
        img_data = cv2.imdecode(np.fromfile(file_path + file, dtype=np.uint8), -1)  # 可读取中文文件名
        try:
            gray = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
        except Exception as e:
            gray = img_data     # 处理文件本身已经为灰值图的情况



        # # 固定阈值二值化
        # ret, img_binary = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)      # 阈值硬编码为175，之后可以更改

        # 自适应阈值二值化
        # 图像取非（将黑白像素颠倒，若不进行此操作，则无法检测到黑色文本像素的连通区域）
        grayNot = cv2.bitwise_not(gray)
        img_binary = cv2.adaptiveThreshold(grayNot, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        # 图像取非
        img_binary = cv2.bitwise_not(img_binary)

        cv2.imencode('.jpg', img_binary)[1].tofile(save_path + str(file).replace(".*", "") + "_binary" + ".png")   # 可存储中文文件名
        # 打印进度
        count += 1
        count_num.set(count)
        process.set("Processing: " + str(count_num.get()) + " / " + str(file_num.get()))
        frame_pre.update()
        print("Finish: " + str(count) + " / " + str(file_len))
    process.set("Finished!")
    frame_pre.update()
    print("Finished!")

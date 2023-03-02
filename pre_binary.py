import os
import numpy as np
import imghdr
import cv2


# 图片二值化
def pre_binary(pre_select_dir_path, pre_save_dir_path, frame_pre, process, process_end, file_num, count_num):
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

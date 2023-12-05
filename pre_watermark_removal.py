import numpy as np
import cv2
from PIL import Image
# 图片二值化
import load_files



def pre_watermark_removal(pre_select_dir_path, pre_save_dir_path, frame_pre, process, file_num, count_num):
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

        black = 108
        white = 164
        threshold = 0.02

        if white > 255:
            white = 255
        if black < 0:
            black = 0
        if black >= white:
            black = white - 2
        img_array = np.array(img, dtype=int)
        cRate = -(white - black) / 255.0 * threshold  # 调节阈值
        rgb_diff = img_array - black
        rgb_diff = np.maximum(rgb_diff, 0)
        img_array = rgb_diff * cRate
        img_array = np.around(img_array, 0)
        img_array = img_array.astype(int)
        img_array = img_array.astype('uint8')
        img = Image.fromarray(img_array)
        img = np.array(img)
        cv2.imencode('.jpg', img)[1].tofile(save_path + str(file).replace(".*", "") + "_removed" + ".png")  # 可存储中文文件名


        # 打印进度
        count += 1
        count_num.set(count)
        process.set("Processing: " + str(count_num.get()) + " / " + str(file_num.get()))
        frame_pre.update()
        print("Finish: " + str(count) + " / " + str(file_len))
    process.set("Finished!")
    frame_pre.update()
    print("Finished!")

import load_files
import cv2
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation as inter


# 仿射变换
def rotate(img, angle, center=None, scale=1.0):
    (w, h) = img.shape[0:2]
    if center is None:
        center = (w//2, h//2)
    wrapMat = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img, wrapMat, (h, w), borderValue=(255, 255,255))


# 通过最小外接矩形框求校正角度
def pre_skew(pre_select_dir_path, pre_save_dir_path, frame_pre, process, file_num, count_num):
    process.set("")
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
        img_data = cv2.imdecode(np.fromfile(file_path+file, dtype=np.uint8), -1)  # 可读取中文文件名
        # 灰度化
        try:
            gray = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
        except Exception as e:
            gray = img_data     # 处理文件本身已经为灰值图的情况
        # 图像取非
        grayNot = cv2.bitwise_not(gray)
        # 二值化
        img_binary = cv2.threshold(grayNot, 175, 255, cv2.THRESH_BINARY, )[1]

        # 获得有文本区域的点集，求最小外接矩形框，并返回旋转角度
        coords = np.column_stack(np.where(img_binary > 0))
        box = cv2.minAreaRect(coords)
        angle = box[-1]
        center = box[0]
        size = box[1]
        if angle < -45:
            angle = -(angle + 90)
        if angle > 45:
            angle = -(angle - 90)
        else:
            angle = -angle

        # # 绘制矩形
        # box = cv2.boxPoints(cv2.minAreaRect(coords))
        # box = np.int0(box)
        # print(box)
        # img_box = img_data.copy()
        # box = np.flip(box, axis=1)    # 翻转矩阵（颠倒坐标xy顺序）
        # print(box)
        # #img_box = cv2.drawContours(img_box, box, 0, 255, 10)
        # img_box = cv2.polylines(img_box, [box], True, (0, 255, 10))
        #
        # print(img_box.shape)
        # cv2.imencode('.jpg', img_box)[1].tofile(save_path + str(file).replace(".*", "") + "_box" + ".png")

        # 仿射变换，校正图像
        img_corrected = rotate(img_data, angle)

        ## 存储校正后的图片
        #cv2.imencode('.jpg', img_corrected)[1].tofile(save_path + str(file).replace(".*", "") + "_corrected" + ".png")  # 可存储中文文件名

        # 根据外接矩形框的大小裁剪图像
        size_0 = size[0]
        size_1 = size[1]
        if size_0 >= size_1:
            h = size_0 + 200     # 稍微增大长宽，与防过度裁剪而漏字
            w = size_1 + 200     # 稍微增大长宽，与防过度裁剪而漏字
        else:
            w = size_0 + 200
            h = size_1 + 200
        size = (w, h)   # 倒转w与h的顺序
        size = np.int0(size)

        center_0 = center[0]
        center_1 = center[1]
        center = (center_1, center_0)   # 倒转坐标xy的顺序

        img_cropped = cv2.getRectSubPix(img_corrected, size, center)

        ## 存储裁剪后的图片
        #cv2.imencode('.jpg', img_cropped)[1].tofile(save_path + str(file).replace(".*", "") + "_cropped" + ".png")  # 可存储中文文件名

        # 给裁剪后的图片增加一个稍大的白色边框，防止裁剪过度而漏掉字符
        top, bottom, left, right = 50, 50, 50, 50
        borderType = cv2.BORDER_CONSTANT
        img_bordered = cv2.copyMakeBorder(img_cropped, top, bottom, left, right, borderType, value=(255,255,255))

        # 存储增加边框后的图片
        cv2.imencode('.jpg', img_bordered)[1].tofile(save_path + str(file).replace(".*", "") + "_cropped" + ".png")  # 可存储中文文件名

        # 打印进度
        count += 1
        count_num.set(count)
        process.set("Processing: " + str(count_num.get()) + " / " + str(file_num.get()))
        frame_pre.update()
        print("Finish: " + str(count) + " / " + str(file_len))
    process.set("Finished!")
    frame_pre.update()
    print("Finished!")


# 垂直投影法，效果不佳，放弃
def find_score(img, angle):
    data = inter.rotate(img, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

# 垂直投影法，效果不佳，放弃
def pre_skew_2(pre_select_dir_path, pre_save_dir_path, frame_pre, process, file_num, count_num):
    process.set("")
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

        # 对图片进行二值化
        img_data = cv2.imdecode(np.fromfile(file_path + file, dtype=np.uint8), -1)  # 可读取中文文件名
        try:
            gray = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
        except Exception as e:
            gray = img_data     # 处理文件本身已经为灰值图的情况
        ret, img_binary = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)      # 阈值硬编码为175，之后可以改

        # 寻找最佳校正角度
        limit = 10   # 设置校正角度限制
        delta = 1   # 设置寻找角度的粒度
        angles = np.arange(-limit, limit+delta, delta)    # 生成所要尝试的角度列表
        scores = []
        for angle in angles:
            hist, score = find_score(img_binary, angle)
            scores.append(score)
        best_score = max(scores)
        best_angle = angles[scores.index(best_score)]

        # 根据最佳角度进行校正
        data = inter.rotate(img_binary, best_angle, reshape=False, order=0)
        # img_corrected = im.fromarray((255 * data).astype("uint8")).convert("RGB")

        # 存储校正后的图片
        cv2.imencode('.jpg', data)[1].tofile(save_path + str(file).replace(".*", "") + "_corrected" + ".png")  # 可存储中文文件名

        # 打印进度
        count += 1
        count_num.set(count)
        process.set("Processing: " + str(count_num.get()) + " / " + str(file_num.get()))
        frame_pre.update()
        print("Finish: " + str(count) + " / " + str(file_len))
    process.set("Finished!")
    frame_pre.update()
    print("Finished!")
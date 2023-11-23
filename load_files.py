import os
import imghdr


def load_img(select_dir_path):
    # 略过非图片文件，将图片文件存入新的列表
    filelist = os.listdir(select_dir_path.get())
    new_filelist = []
    for file in filelist:
        file_path = select_dir_path.get() + '/' + file
        img_type = {'jpg', 'jpeg', 'bmp', 'png', 'tif', 'tiff'}
        if imghdr.what(file_path) not in img_type:
            continue
        new_filelist.append(file)
    return new_filelist

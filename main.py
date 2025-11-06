import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import webbrowser
import time
import threading

import pre_denoise
import pre_binary
import pre_skew
import pre_split
import pre_split_2_lines
import pre_split_without_line
import pre_black_edge
import pre_upper_line_removal
import pre_watermark_removal
import pre_rotate
import ocr_baidu
# import ocr_paddle


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


# Github链接
def open_url():
    webbrowser.open("https://github.com/yukiyuqichen/OCR-Toolkit")


# 前处理
def preprocess():
    if pre_option.get() == "Denoise 去噪":
        pre_denoise.pre_denoise(pre_select_dir_path, pre_save_dir_path, frame_pre, process, file_num, count_num)
    if pre_option.get() == "Binarization 二值化":
        pre_binary.pre_binary(pre_select_dir_path, pre_save_dir_path, frame_pre, process, file_num, count_num)
    if pre_option.get() == "Skew Correction 倾斜校正":
        pre_skew.pre_skew(pre_select_dir_path, pre_save_dir_path, frame_pre, process, file_num, count_num)
    if pre_option.get() == "Split 根据中间1条界栏分割图片":
        pre_split.pre_split(pre_select_dir_path, pre_save_dir_path, frame_pre, process, file_num, count_num)
    if pre_option.get() == "Split 根据中间2条界栏分割图片":
        pre_split_2_lines.pre_split(pre_select_dir_path, pre_save_dir_path, frame_pre, process, file_num, count_num)
    if pre_option.get() == "Split 根据中间空白分割图片":
        pre_split_without_line.pre_split_without_line(pre_select_dir_path, pre_save_dir_path, frame_pre, process, file_num, count_num)
    if pre_option.get() == "Black Edge 去黑边":
        pre_black_edge.pre_black_edge(pre_select_dir_path, pre_save_dir_path, frame_pre, process, file_num, count_num)
    if pre_option.get() == "Upper Line 去除上方界栏":
        pre_upper_line_removal.pre_upper_line_removal(pre_select_dir_path, pre_save_dir_path, frame_pre, process, file_num, count_num)
    if pre_option.get() == "Rotate 旋转图片":
        pre_rotate.pre_rotate(pre_select_dir_path, pre_save_dir_path, frame_pre, process, file_num, count_num)
    if pre_option.get() == "Watermark 去水印":
        pre_watermark_removal.pre_watermark_removal(pre_select_dir_path, pre_save_dir_path, frame_pre, process, file_num, count_num)


# OCR
def ocr():
    # if ocr_option.get() == "Offline":
    #     ocr_paddle.ocr_paddle(frame_ocr, ocr_select_dir_path, ocr_save_dir_path, process, file_num, count_num)
    if ocr_option.get() == "BaiduAI":
        ocr_baidu.ocr_baidu_parser(frame_ocr, ocr_select_dir_path, ocr_save_dir_path, api_key, secret_key, token, process, file_num, count_num)


# 多线程防卡顿
def thread_it(func, *args):
    """将函数打包进线程"""
    # 创建
    t = threading.Thread(target=func, args=args)
    # 守护
    t.daemon = True
    # 启动
    t.start()
    # 阻塞--卡死界面！
    # t.join()



# 创建主窗口
root_window = tk.Tk()
# 设置主窗口标题
root_window.title('OCR Toolkit')
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
com_pre = ttk.Combobox(frame_pre, textvariable=pre_option)
com_pre["value"] = ("Denoise 去噪", "Binarization 二值化", "Skew Correction 倾斜校正", "Split 根据中间1条界栏分割图片", "Split 根据中间2条界栏分割图片", "Split 根据中间空白分割图片", "Black Edge 去黑边", "Upper Line 去除上方界栏", "Watermark 去水印", "Rotate 旋转图片")
com_pre.current(0)
com_pre.grid(row=1, column=3)
# 路径选择组件
tk.Button(frame_pre, text="Image Path", command=pre_select_dir).grid(row=2, column=2)
tk.Entry(frame_pre, textvariable=pre_select_dir_path).grid(row=2, column=3)
tk.Button(frame_pre, text="Output Path", command=pre_save_dir).grid(row=3, column=2)
tk.Entry(frame_pre, textvariable=pre_save_dir_path).grid(row=3, column=3)
# 开始运行按钮
tk.Button(frame_pre, text="Start", command=lambda: thread_it(preprocess)).grid(row=4, column=4)
# 作装饰的空label
tk.Label(frame_pre, text="").grid(row=5, column=1)
# 进度条显示
tk.Label(frame_pre, textvariable=process).grid(row=6, column=1, columnspan=3)


# 设置frame_ocr中的变量
# 路径变量
ocr_select_dir_path = tk.StringVar()    # 准备OCR的图片文件夹所在路径
ocr_save_dir_path = tk.StringVar()      # OCR完成后的文本文件存储路径
# Token变量
api_key = tk.StringVar()
secret_key = tk.StringVar()
# 选项变量
ocr_option = tk.StringVar()

# 设置frame_ocr中的组件
# 作装饰的空label
tk.Label(frame_ocr, text="      ").grid(row=0, column=1)
# 下拉菜单
tk.Label(frame_ocr, text="Option: ").grid(row=1, column=2)
com_ocr = ttk.Combobox(frame_ocr, textvariable=ocr_option)
# com_ocr["value"] = ("Offline", "BaiduAI")
com_ocr["value"] = ("BaiduAI")
com_ocr.current(0)
com_ocr.grid(row=1, column=3)
# api_key输入组件
tk.Label(frame_ocr, text="API_KEY: ").grid(row=2, column=2)
tk.Entry(frame_ocr, textvariable=api_key).grid(row=2, column=3)
tk.Label(frame_ocr, text="SECRET_KEY: ").grid(row=3, column=2)
tk.Entry(frame_ocr, textvariable=secret_key).grid(row=3, column=3)
# 路径选择组件
tk.Button(frame_ocr, text="Image Path", command=ocr_select_dir).grid(row=4, column=2)
tk.Entry(frame_ocr, textvariable=ocr_select_dir_path).grid(row=4, column=3)
tk.Button(frame_ocr, text="Output Path", command=ocr_save_dir).grid(row=5, column=2)
tk.Entry(frame_ocr, textvariable=ocr_save_dir_path).grid(row=5, column=3)
# 开始运行按钮
tk.Button(frame_ocr, text="Start", command=lambda: thread_it(ocr)).grid(row=6, column=4)
# 作装饰的空label
tk.Label(frame_ocr, text="").grid(row=7, column=1)
# 进度条显示
tk.Label(frame_ocr, textvariable=token).grid(row=8, column=1, columnspan=3)
tk.Label(frame_ocr, textvariable=process).grid(row=9, column=1, columnspan=3)


# 设置frame_about中的组件
tk.Label(frame_about, text="                    ").grid(row=0, column=0)
tk.Label(frame_about, text="For CBDB group").grid(row=1, column=1)
tk.Label(frame_about, text="Developed by: Yuqi Chen").grid(row=2, column=1)
tk.Label(frame_about, text="Email: cyq0722@pku.edu.cn").grid(row=3, column=1)
tk.Label(frame_about, text="Thanks to: Hongsu Wang").grid(row=4, column=1)
tk.Label(frame_about, text="                    ").grid(row=5, column=0)
tk.Label(frame_about, text="Icon drawn by: DALL·E 2").grid(row=6, column=1)
tk.Label(frame_about, text="Version: 2023.03.02").grid(row=7, column=1)
tk.Label(frame_about, text="                ").grid(row=8, column=0)
tk.Button(frame_about, text="OCR-Toolkit on Github", command=open_url).grid(row=9, column=1)


# 进入主循环，显示主窗口
root_window.mainloop()




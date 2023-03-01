# OCR-Toolkit
A cute toolkit for OCR, including preprocessing and API use.  

一只小小的OCR工具箱，包括图像预处理和API接口调用等功能。

![image](https://github.com/yukiyuqichen/OCR-Toolkit/blob/main/elephant.ico)

# Download
[OCR Toolkit 2023.2.28](https://github.com/yukiyuqichen/OCR-Toolkit/releases)

# Preprocessing
## 1. Binary  
Denoise the image with binarization based on thresholding.  

对图像进行基于阈值分割的二值化，简单去噪。

<p float="left">
  <img src="https://github.com/yukiyuqichen/OCR-Toolkit/blob/main/examples/binary_before.png" width="300" />
  <img src="https://github.com/yukiyuqichen/OCR-Toolkit/blob/main/examples/binary_after.png" width="300" />
</p>

## 2. Split
Detect the middle line with Hough transform algorithm and segment the image into two parts. It might come in handy when handling documents like dictionary.  

通过霍夫变换检测中间界栏，根据界栏对图像进行分割，适用于词典等版式的文档。

<p float="left">
  <img src="https://github.com/yukiyuqichen/OCR-Toolkit/blob/main/examples/split.png" width="300" />
  <img src="https://github.com/yukiyuqichen/OCR-Toolkit/blob/main/examples/split_line.png" width="300" />
</p>
<p float="left">
  <img src="https://github.com/yukiyuqichen/OCR-Toolkit/blob/main/examples/split_left.png" width="300" /> 
  <img src="https://github.com/yukiyuqichen/OCR-Toolkit/blob/main/examples/split_right.png" width="300" />
</p>

# OCR API  

## 1. Offline: PaddleOCR
Use PaddleOCR models to get the result of OCR.

<p float="left">
  <img src="https://github.com/yukiyuqichen/OCR-Toolkit/blob/main/examples/ocr_paddle_before.png" width="300" /> 
  <img src="https://github.com/yukiyuqichen/OCR-Toolkit/blob/main/examples/ocr_paddle_after.png" width="600" />
</p>


## 2. Online: Baidu API
Use api of Baidu AI to get the result of OCR and parse it. The text file and probability file are created and aligned.  
Users need to provide the API_KEY and SECRET_KEY.  
More APIs are going to be included.  

使用Baidu AI高精度文字识别的API接口，对图像进行OCR，并将返回的结果解析为对齐的文本文件和置信度文件。  
用户需自行输入API_KEY和SECRET_KEY。  
更多接口扩充中。

# OCR-Toolkit
A cute toolkit for OCR, including image preprocessing and text recognition. Works out of the box.  

一只小小的OCR工具箱，包括图像预处理和文字识别等功能，开箱即用。  

<img src="https://github.com/yukiyuqichen/OCR-Toolkit/blob/main/elephant.ico" width="300" />

# Download
The exe file can be downloaded: [OCR Toolkit 2023.03.02 new](https://github.com/yukiyuqichen/OCR-Toolkit/releases/tag/2023.03.02-new)

<p float="left">
<img src="https://github.com/yukiyuqichen/OCR-Toolkit/blob/main/examples/UI_1.png" width="300" />
<img src="https://github.com/yukiyuqichen/OCR-Toolkit/blob/main/examples/UI_2.png" width="300" />
</p>

# 1. Preprocessing
## 1.1 Binary  
Denoise the image with Binarization Thresholding.  

对图像进行基于阈值分割的二值化，简单去噪。

<p float="left">
  <img src="https://github.com/yukiyuqichen/OCR-Toolkit/blob/main/examples/binary_before.png" width="250" />
  <img src="https://github.com/yukiyuqichen/OCR-Toolkit/blob/main/examples/binary_after.png" width="250" />
</p>

## 1.2 Split
Detect the middle line with Hough transform algorithm and segment the image into two parts. It might come in handy when handling documents like dictionary.  

通过霍夫变换检测中间界栏，根据界栏对图像进行分割，适用于词典等版式的文档。

<p float="left">
  <img src="https://github.com/yukiyuqichen/OCR-Toolkit/blob/main/examples/split.png" width="250" />
  <img src="https://github.com/yukiyuqichen/OCR-Toolkit/blob/main/examples/split_line.png" width="250" />
</p>
<p float="left">
  <img src="https://github.com/yukiyuqichen/OCR-Toolkit/blob/main/examples/split_left.png" width="250" /> 
  <img src="https://github.com/yukiyuqichen/OCR-Toolkit/blob/main/examples/split_right.png" width="250" />
</p>

# 2. OCR  

## 2.1 Offline: PaddleOCR
Use [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) models to get the result of OCR.  
No KEY is needed.
The result will be saved as a structured csv file.  
在本地部署PaddleOCR模型，对图像进行OCR，并将结果存储为结构化的csv文件。  

<p float="left">
  <img src="https://github.com/yukiyuqichen/OCR-Toolkit/blob/main/examples/ocr_paddle_before.png" width="250" /> 
  <img src="https://github.com/yukiyuqichen/OCR-Toolkit/blob/main/examples/ocr_paddle_after.png" width="500" />
</p>

<img src="https://github.com/yukiyuqichen/OCR-Toolkit/blob/main/examples/ocr_paddle_result.png" width="550" />

## 2.2 Online: Baidu API
Use api of [Baidu AI](https://ai.baidu.com/tech/ocr/general) to get the result of OCR and parse it. The result will be saved as a structured csv file.  
Users need to provide the API_KEY and SECRET_KEY.  
More APIs are going to be included.  

使用Baidu AI高精度文字识别的API接口，对图像进行OCR，并将结果存储为结构化的csv文件。  
用户需自行输入API_KEY和SECRET_KEY。  
更多接口扩充中。

<img src="https://github.com/yukiyuqichen/OCR-Toolkit/blob/main/examples/ocr_baidu_result.png" width="750" />

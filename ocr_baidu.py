import os
import imghdr
import base64
import requests
import urllib


# 接入百度API进行OCR
def get_access_token(api_key, secret_key):
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": api_key.get(), "client_secret": secret_key.get()}
    return str(requests.post(url, params=params).json().get("access_token"))


def ocr_baidu(image, frame_ocr, token, api_key, secret_key):
    with open(image, 'rb') as f:
        image_data = f.read()
        base64_data = base64.b64encode(image_data)  # 对图片进行base64编码
        base64_data_url = urllib.parse.quote(base64_data)  # 对base64编码进行url编码
    url = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic?access_token=" + get_access_token(api_key, secret_key)    # 高精度不含位置版
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


def ocr_baidu_parser(frame_ocr, ocr_select_dir_path, ocr_save_dir_path, api_key, secret_key, token, process, process_end, file_num, count_num):
    token.set("")
    process.set("")
    process_end.set("")
    frame_ocr.update()
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
    save_filename = []
    save_text = []
    save_score_ave = []
    save_score_min = []
    save_score_var = []
    for file in new_filelist:
        file_path = ocr_select_dir_path.get() + '/' + file
        json = ocr_baidu(file_path, frame_ocr, token, api_key, secret_key)
        text = json["words_result"]
        for i in text:
            save_filename.append(str(file))
            save_text.append(i['words'])
            save_score_ave.append(i['probability']['average'])
            save_score_min.append(i['probability']['min'])
            save_score_var.append(i['probability']['variance'])
        with open(ocr_save_dir_path.get() + '/' + 'result.csv', 'w', encoding='utf-8-sig', errors='ignore') as f:
            f.write("filename,text,score_ave,score_min,score_var" + "\n")
            for i in range(len(save_filename)):
                f.write(save_filename[i] + ",")
                f.write(save_text[i] + ",")
                f.write(str(save_score_ave[i]) + ",")
                f.write(str(save_score_min[i]) + ",")
                f.write(str(save_score_var[i]) + "\n")
        # 打印进度
        count += 1
        count_num.set(count)
        process.set("Processing: " + str(count_num.get()) + " / " + str(file_num.get()))
        frame_ocr.update()
        print("Finish: " + str(count) + " / " + str(file_len))
    process_end.set("Finished!")
    frame_ocr.update()
    print("Finished!")

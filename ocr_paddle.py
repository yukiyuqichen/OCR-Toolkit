import os
import sys
import PaddleOCR.tools.infer.utility as utility
import PaddleOCR.tools.infer.predict_system as predict


# 在本地部署PaddleOCR
def ocr_paddle(frame_ocr, ocr_select_dir_path, ocr_save_dir_path, process, file_num, count_num):
    args = utility.parse_args()
    args.image_dir = ocr_select_dir_path.get()
    args.draw_img_save_dir = ocr_save_dir_path.get()
    args.crop_res_save_dir = ocr_save_dir_path.get()
    args.save_log_path = ocr_save_dir_path.get()
    args.det_model_dir = "./models/ch_PP-OCRv3_det_infer/"
    args.rec_model_dir = "./models/ch_PP-OCRv3_rec_infer/"
    args.cls_model_dir = "./models/ch_ppocr_mobile_v2.0_cls_infer/"
    try:
        predict.main(args, frame_ocr, process, file_num, count_num)
    except Exception as e:
        process.set(str(e))
        frame_ocr.update()

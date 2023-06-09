from models.ocr.ppocr_inference import PaddleOCR_Inference
import cv2 as cv
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

use_gpu = False
det_model_dir = r"weights\det\det_r50_db++_icdar15_train_3"
det_db_thresh = 0.5
det_db_box_thresh = 0.6
det_algorithm = "DB++"
det_limit_side_len = 2048
det_limit_type = "max"
det_db_score_mode = "slow"
                
rec_model_dir = r"weights\rec\r45_abinet_2"
rec_algorithm = "ABINet"
rec_image_shape = "3,32,128"
rec_drop_score = 0.001
rec_char_dict_path = r"utils\vi_vietnam_new.txt"
rec_use_space_char = True
            
task="ocr"

ppocr = PaddleOCR_Inference(
                use_gpu, 
                det_model_dir, det_db_thresh, det_db_box_thresh, det_algorithm, det_limit_side_len, det_limit_type, det_db_score_mode,
                rec_model_dir, rec_algorithm, rec_image_shape, rec_drop_score, rec_char_dict_path, rec_use_space_char,
                task)

img_path = r"test_img\004.png"
img = cv.imread(img_path)
boxes, texts = ppocr(img)

print(f"Boxes:{boxes}")
print(f"Texts:{texts}")
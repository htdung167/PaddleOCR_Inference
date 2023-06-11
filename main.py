import cv2 as cv
import os
from PaddleOCR.ppocr_inference import PaddleOCR_Inference
from PaddleOCR.ppocr_inference import visualize_ocr
#--Parameters
use_gpu = False

det_model_dir = r"weights\det\det_r50_db++_icdar15_train_3" # Weights
det_db_thresh = 0.5
det_db_box_thresh = 0.6
det_algorithm = "DB++"
det_limit_side_len = 2048
det_limit_type = "max"
det_db_score_mode = "slow"
                
rec_model_dir = r"weights\rec\r45_abinet_2" # Weights
rec_algorithm = "ABINet"
rec_image_shape = "3,32,128"
rec_drop_score = 0.001
rec_char_dict_path = r"PaddleOCR\vi_vietnam_new.txt" # Dictionary
rec_use_space_char = True
            
task="ocr"

#--Initial
ppocr = PaddleOCR_Inference(
                use_gpu, 
                det_model_dir, det_db_thresh, det_db_box_thresh, det_algorithm, det_limit_side_len, det_limit_type, det_db_score_mode,
                rec_model_dir, rec_algorithm, rec_image_shape, rec_drop_score, rec_char_dict_path, rec_use_space_char,
                task)

#--Inference
img_path = r"test_img\005.png"

img = cv.imread(img_path)
boxes, texts = ppocr(img)

# Log
print(f"Boxes:{boxes}")
print(f"Texts:{texts}")

# Save
name = img_path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] 
res = visualize_ocr(img=img, lst_box=boxes, lst_text=texts, font_path="PaddleOCR/font/arial.ttf")
cv.imwrite(os.path.join("experiments", name), res)


from .PaddleOCR.tools.infer.predict_det import TextDetector
from .PaddleOCR.tools.infer.predict_rec import TextRecognizer
import numpy as np
import cv2 as cv
import copy
from models.ocr.args_alt import ArgsAlt

class PaddleOCR_Inference():
    def __init__(self, use_gpu, 
                 det_model_dir, det_db_thresh, det_db_box_thresh, det_algorithm, det_limit_side_len, det_limit_type, det_db_score_mode,
                 rec_model_dir, rec_algorithm, rec_image_shape, rec_drop_score, rec_char_dict_path, rec_use_space_char,
                 task="ocr"):
        
        assert task in ["det", "rec", "ocr"]
        self.use_gpu = use_gpu
        self.detector = None
        self.recognizer = None
        self.task = task

        if task in ["det", "ocr"]:
            self.det_model_dir = det_model_dir
            self.det_db_thresh = det_db_thresh
            self.det_db_score_mode = det_db_score_mode
            self.det_db_box_thresh = det_db_box_thresh
            self.det_algorithm = det_algorithm
            self.det_limit_side_len = det_limit_side_len
            self.det_limit_type = det_limit_type
            self.detector = self.load_text_detector()

        if task in ["rec", "ocr"]:
            self.rec_model_dir = rec_model_dir
            self.rec_algorithm = rec_algorithm 
            self.rec_image_shape = rec_image_shape 
            self.rec_drop_score = rec_drop_score 
            self.rec_char_dict_path = rec_char_dict_path
            self.rec_use_space_char = rec_use_space_char
            self.recognizer = self.load_text_recognizer()

    def load_text_detector(self):
        # args = utility.parse_args()
        args = ArgsAlt()

        args.use_gpu = self.use_gpu

        args.det_algorithm = self.det_algorithm
        args.det_model_dir = self.det_model_dir
        args.det_db_thresh = self.det_db_thresh
        args.det_db_box_thresh = self.det_db_box_thresh
        args.det_limit_side_len = self.det_limit_side_len
        args.det_limit_type = self.det_limit_type
        args.det_db_score_mode = self.det_db_score_mode
        
        text_detector = TextDetector(args)
        return text_detector
    
    def load_text_recognizer(self):
        # args = utility.parse_args()
        args = ArgsAlt()

        args.use_gpu = self.use_gpu

        args.rec_model_dir = self.rec_model_dir
        args.rec_algorithm = self.rec_algorithm
        args.rec_image_shape = self.rec_image_shape  
        args.drop_score = self.rec_drop_score
        args.rec_char_dict_path = self.rec_char_dict_path 
        args.use_space_char = self.rec_use_space_char
        
        text_recognizer = TextRecognizer(args)
        return text_recognizer

    def detect(self, img):
        dt_boxes, _ = self.detector(img)
        dt_boxes = sorted_boxes(dt_boxes)
        new_dt_boxes = []
        for points in dt_boxes:
            new_points = convert_quad_to_rect(points=points)
            new_dt_boxes.append(new_points)
        return new_dt_boxes

    def recognize(self, lst_img):
        rec_res, _ = self.recognizer(lst_img)
        return rec_res

    def ocr(self, img):
        ori_im = img.copy()

        dt_boxes = self.detect(img=img)

        img_crop_list = []
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        
        rec_res = self.recognize(img_crop_list)

        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.rec_drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(text)

        return filter_boxes, filter_rec_res
            

    def __call__(self, img):
        if self.task == "det":
            return self.detect(img), None
        elif self.task == "rec":
            # img is list image
            return None, self.recognize(img)
        elif self.task == "ocr":
            return self.ocr(img)
        else:
            return None, None

        
def convert_quad_to_rect(points):
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    return np.array([left, top, right, bottom])
    
def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes

def get_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    assert len(points) == 4, "shape of points must be 4*2"

    left, top, right, bottom = points
    new_points = np.float32([[left, top], [right, top], [right, bottom], [left, bottom]])

    img_crop_width = int(
        max(
            np.linalg.norm(new_points[0] - new_points[1]),
            np.linalg.norm(new_points[2] - new_points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(new_points[0] - new_points[3]),
            np.linalg.norm(new_points[1] - new_points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    
    M = cv.getPerspectiveTransform(new_points, pts_std)
    dst_img = cv.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv.BORDER_REPLICATE,
        flags=cv.INTER_CUBIC)
    return dst_img

def order_points(points):
    points = np.array(points.copy())
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


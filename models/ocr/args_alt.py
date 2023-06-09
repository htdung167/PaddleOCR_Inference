
class ArgsAlt():
    def __init__(self):
        # params for prediction engine
        self.use_gpu = True
        self.use_xpu = False
        self.use_npu = False
        self.ir_optim = False
        self.use_tensorrt = False
        self.min_subgraph_size = 15
        self.precision = "fp32"
        self.gpu_mem = 500
        self.gpu_id = 0

        # params for text detector
        self.image_dir = ""
        self.page_num = 0
        self.det_algorithm = "DB"
        self.det_model_dir = ""
        self.det_limit_side_len = 960
        self.det_limit_type = "max"
        self.det_box_type = "quad"
    
        # DB parmas
        self.det_db_thresh = 0.3
        self.det_db_box_thresh = 0.6
        self.det_db_unclip_ratio = 1.5
        self.max_batch_size = 10
        self.use_dilation = False
        self.det_db_score_mode = "fast"

        # EAST parmas
        self.det_east_score_thresh = 0.8
        self.det_east_cover_thresh = 0.1
        self.det_east_nms_thresh = 0.2

        # SAST parmas
        self.det_sast_score_thresh = 0.5
        self.det_sast_nms_thresh = 0.2

        # PSE parmas
        self.det_pse_thresh = 0
        self.det_pse_box_thresh = 0.85
        self.det_pse_min_area = 16
        self.det_pse_scale = 1

        # FCE parmas
        self.scales = [8, 16, 32]
        self.alpha = 1.0
        self.beta = 1.0
        self.fourier_degree = 5
        
        # params for text recognizer
        self.rec_algorithm = 'SVTR_LCNet'
        self.rec_model_dir = ""
        self.rec_image_inverse = True
        self.rec_image_shape = "3, 48, 320"
        self.rec_batch_num = 6
        self.max_text_length = 25
        self.rec_char_dict_path = "./ppocr/utils/ppocr_keys_v1.txt"
        self.use_space_char = True
        self.vis_font_path = "./doc/fonts/simfang.ttf"
        self.drop_score = 0.5

        # params for e2e
        self.e2e_algorithm = 'PGNet'
        self.e2e_model_dir = ''
        self.e2e_limit_side_len = 768
        self.e2e_limit_type = 'max'
        self.e2e_pgnet_score_thresh = 0.5
        self.e2e_char_dict_path = "./ppocr/utils/ic15_dict.txt"
        self.e2e_pgnet_valid_set = 'totaltext'
        self.e2e_pgnet_mode = 'fast'

        # params for text classifier
        self.use_angle_cls = False
        self.cls_model_dir = ''
        self.cls_image_shape = "3, 48, 192"
        self.label_list = ['0', '180']
        self.cls_batch_num = 6
        self.cls_thresh = 0.9
        self.enable_mkldnn = False
        self.cpu_threads = 10
        self.use_pdserving = False
        self.warmup = False

        # SR parmas
        self.sr_model_dir = ''
        self.sr_image_shape = "3, 32, 128"
        self.sr_batch_num = 1
        self.draw_img_save_dir = "./inference_results"
        self.save_crop_res = False
        self.crop_res_save_dir = "./output"

        # multi-process
        self.use_mp = False
        self.total_process_num = 1
        self.process_id = 0
        self.benchmark = False
        self.save_log_path = "./log_output/"
        self.show_log = True
        self.use_onnx = False
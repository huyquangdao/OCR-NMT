from easydict import EasyDict

DETECTION_CONFIG = EasyDict()

DETECTION_CONFIG.MODEL_PATH = 'OCR/weights/craft_mlt_25k.pth'
DETECTION_CONFIG.TEXT_THRESHOLD = 0.7
DETECTION_CONFIG.LOW_TEXT = 0.4
DETECTION_CONFIG.LINK_THRESHOLD = 0.4
DETECTION_CONFIG.CUDA = False
DETECTION_CONFIG.CANVAS_SIZE = 1280
DETECTION_CONFIG.MAG_RATIO = 1.5
DETECTION_CONFIG.POLY = False
DETECTION_CONFIG.SHOW_TIME = True


RECOGNITION_CONFIG = EasyDict()

RECOGNITION_CONFIG.Transformation = 'TPS'
RECOGNITION_CONFIG.FeatureExtraction = 'ResNet'
RECOGNITION_CONFIG.SequenceModeling = 'BiLSTM'
RECOGNITION_CONFIG.Prediction = 'Attn'
RECOGNITION_CONFIG.num_fiducial= 20
RECOGNITION_CONFIG.input_channel = 1
RECOGNITION_CONFIG.output_channel= 512
RECOGNITION_CONFIG.hidden_size = 256
RECOGNITION_CONFIG.batch_max_length = 25
RECOGNITION_CONFIG.imgH= 32
RECOGNITION_CONFIG.imgW = 100

RECOGNITION_CONFIG.rgb = False
RECOGNITION_CONFIG.character = '0123456789abcdefghijklmnopqrstuvwxyz'
RECOGNITION_CONFIG.sensitive = True
RECOGNITION_CONFIG.saved_model = 'OCR/weights/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth'
RECOGNITION_CONFIG.batch_size = 1
RECOGNITION_CONFIG.workers = 1
RECOGNITION_CONFIG.CUDA = False

RECOGNITION_CONFIG.PAD = False

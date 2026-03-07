"""
Configuration management for DA-SAM
"""
import os
from collections import OrderedDict
from pprint import pprint


class Config:
    """Configuration class for DA-SAM training and inference"""

    def __init__(self):
        # Project paths (modify these for your environment)
        self.ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(self.ROOT, "data")
        self.OUTPUT_DIR = os.path.join(self.ROOT, "experiments")
        self.CHECKPOINT_DIR = os.path.join(self.ROOT, "checkpoints")

        # SAM checkpoint
        self.SAM_CKPT = os.path.join(self.CHECKPOINT_DIR, "sam_vit_b_01ec64.pth")
        self.MODEL_TYPE = "vit_b"

        # Data paths
        self.DIR_PROJ = os.path.join(self.DATA_DIR, "proj_img")
        self.DIR_EQ = os.path.join(self.DATA_DIR, "Eq_proj_img")
        self.DIR_LABEL = os.path.join(self.DATA_DIR, "metal_trace_label")
        self.DIR_PROJ_METAL = os.path.join(self.DATA_DIR, "proj_metal")

        # Data splits (sample names)
        self.TRAIN_NAMES = [
            'sample5', 'sample50', 'sample7', 'sample47', 'sample9', 'sample10',
            'sample11', 'sample13', 'sample21', 'sample14', 'sample15', 'sample24',
            'sample40', 'sample43', 'sample26', 'sample38'
        ]
        self.VAL_NAMES = ['sample52', 'sample42', 'sample44']
        self.TEST_NAMES = ['sample46', 'sample8', 'sample6', 'sample48']

        # Model parameters
        self.UNET_BASE_CH = 32
        self.UNET_DEPTH = 4
        self.AIA_CH = 32
        self.MOD_SCALE = 0.5
        self.ADAPTER_DIM = 32
        self.ADAPTER_SCALE = 0.5
        self.LOGIT_SCALE = 10.0

        # Training parameters
        self.IMG_SIZE = 1024
        self.STAGE1_BATCH_SIZE = 2
        self.STAGE1_EPOCHS = 100
        self.STAGE1_PATIENCE = 15
        self.STAGE2_BATCH_SIZE = 1
        self.STAGE2_EPOCHS = 50
        self.STAGE2_PATIENCE = 10
        self.ACCUM_STEPS = 4
        self.NUM_WORKERS = 4
        self.SEED = 42
        self.USE_AMP = False
        self.CLIP_NORM = 1.0

        # Learning rates
        self.LR = 1e-4
        self.LR_AIA = 5e-4
        self.LR_FREQ_ADAPTER = 5e-4
        self.LR_DECODER = 5e-5
        self.WEIGHT_DECAY = 1e-4
        self.WEIGHT_DECAY_STAGE2 = 1e-3

        # Loss weights
        self.LAMBDA_BCE = 0.8
        self.LAMBDA_DICE = 2.0
        self.LAMBDA_AUX = 0.5
        self.TARGET_MOD = 0.3

        # Logging
        self.LOG_EVERY = 50
        self.PRINT_EVERY = 200

        # Inference
        self.THRESHOLD = 0.5

    def load_config(self, config_path):
        """Load configuration from file"""
        print('Config:', config_path)
        if not os.path.exists(config_path):
            print(f"Config file not found: {config_path}")
            return

        with open(config_path, 'r') as f:
            config_list = f.readlines()

        self.config_type_dict = {}

        for each in config_list:
            if len(each) > 4 and each[4] == '#':
                tmp_flag = each.split(' ')[1]
                self.config_type_dict[tmp_flag] = {}

            if each[0] in ['#', '*']:
                continue
            elif each == '\n':
                continue
            else:
                kv = each.strip().split('=')
                if len(kv) < 2:
                    continue
                k = kv[0].strip()
                v = kv[1].strip()
                if v[0] == "'":
                    v = str(v[1:-1])
                elif v[0] == 'T':
                    v = True
                elif v[0] == 'F':
                    v = False
                elif v[0] == 'N':
                    v = None
                elif v[0] == '[':
                    v_split = v[1:-1].split(',')
                    v = [float(each.strip()) for each in v_split]
                elif v.isdigit():
                    v = int(v)
                else:
                    try:
                        v = float(v)
                    except:
                        pass
                self.config_type_dict[tmp_flag][k] = v

        config_type_dict_keys = list(self.config_type_dict.keys())
        for config_type_dict_key in config_type_dict_keys:
            self.__dict__.update(self.config_type_dict[config_type_dict_key])

    def _spec(self, kwargs):
        """Process specific configuration from command line args"""
        print_dict = OrderedDict()
        state_dict = self._state_dict()

        for k, v in kwargs.items():
            setattr(self, k, v)

            if isinstance(v, list) and len(v) > 10:
                continue
            elif k not in state_dict:
                continue
            elif state_dict[k] == v:
                continue
            else:
                print_dict[k] = v

        state_dict = self._state_dict()
        state_dict.pop('config_type_dict', None)
        state_dict_keys = sorted(list(state_dict.keys()))
        save_config_save = OrderedDict()
        for key in state_dict_keys:
            save_config_save[key] = state_dict[key]

        print('======user config========')
        pprint(print_dict)
        print('==========end============')
        return save_config_save

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in self.__dict__.items()
                if not k.startswith('_')}

    def update_paths(self, root=None, data_dir=None, output_dir=None):
        """Update paths for different environments"""
        if root:
            self.ROOT = root
            self.DATA_DIR = os.path.join(root, "data")
            self.OUTPUT_DIR = os.path.join(root, "experiments")
            self.CHECKPOINT_DIR = os.path.join(root, "checkpoints")
            self.SAM_CKPT = os.path.join(self.CHECKPOINT_DIR, "sam_vit_b_01ec64.pth")

        if data_dir:
            self.DATA_DIR = data_dir
            self.DIR_PROJ = os.path.join(data_dir, "proj_img")
            self.DIR_EQ = os.path.join(data_dir, "Eq_proj_img")
            self.DIR_LABEL = os.path.join(data_dir, "metal_trace_label")
            self.DIR_PROJ_METAL = os.path.join(data_dir, "proj_metal")

        if output_dir:
            self.OUTPUT_DIR = output_dir


opt = Config()

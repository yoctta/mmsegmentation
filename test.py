from mmcv.utils import Config, DictAction, get_git_hash
from mmseg.models import build_segmentor
cfg = Config.fromfile('configs/segdiffusion/Analog_bits.py')
model = build_segmentor(cfg.model,train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model=model.cuda()
import torch
x=torch.zeros([2,3,512,512])
y=torch.zeros([2,1,512,512]).long()
torch.distributed.init_process_group('nccl','tcp://127.0.0.1:28888',world_size=1,rank=0)
model.forward_train(x.cuda(),{},y.cuda())
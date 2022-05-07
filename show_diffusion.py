import argparse
import os
import os.path as osp
import shutil
import time
import warnings
from mmcv.image import tensor2imgs
import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import setup_multi_processes
from mmseg.ops import resize
import torch.nn.functional as F
import cv2
import numpy as np
from torch import multiprocessing as mp
config="configs/segdiffusion/upernet_beit-base_512x512_160k_ade20k_20t_kl_loss.py"
checkpoint="pretrain/seg_diff_20t_160k.pth"
out_dir="work_dirs/seg_diff/visualize"
def worker(rank,config,checkpoint,out_dir,world_size):
    torch.cuda.set_device(rank)
    cfg = mmcv.Config.fromfile(config)
    dataset = build_dataset(cfg.data.test)
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=1,
        dist=False)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **cfg.data.get('test_dataloader', {})
    }
    # build the dataloader
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE
    torch.cuda.empty_cache()
    model = revert_sync_batchnorm(model)
    #model = MMDataParallel(model, device_ids=gpu_ids)
    model=model.cuda()
    single_gpu_test(model,data_loader,out_dir,0.5,world_size,rank)

def make_grid(imgs,rows=0,margin=5):
    H,W,C=imgs[0].shape
    li=len(imgs)
    if not rows:
        rows=int(np.floor(np.sqrt(li)))
    cols=int(np.ceil(li/rows))
    pad=np.zeros(((H+margin)*rows-margin,(W+margin)*cols-margin,3),dtype=np.uint8)
    for  i in range(li):
        rs=i//cols
        ls=i%cols
        pad[rs*(H+margin):rs*(H+margin)+H,ls*(W+margin):ls*(W+margin)+W]=imgs[i]
    return pad




def slide_inference(model, img):
    h_stride, w_stride = model.test_cfg.stride
    h_crop, w_crop = model.test_cfg.crop_size
    batch_size, _, h_img, w_img = img.size()
    num_classes = model.num_classes
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    crop_images=[]
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = img[:, :, y1:y2, x1:x2]
            crop_images.append(crop_img)
    def merge_fn(patches):
        preds = torch.zeros((batch_size, num_classes, h_img, w_img),device="cpu")
        count_mat = torch.zeros((batch_size, 1, h_img, w_img),device="cpu")
        i=0
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_seg_logit = patches[i].cpu()
                i+=1
                preds += F.pad(crop_seg_logit,
                                (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        return preds
    return crop_images, merge_fn


def single_gpu_test(model,data_loader,out_dir,opacity=0.5,world_size=1,rank=0):
    model.eval()
    results = []
    dataset = data_loader.dataset
    if rank==0:
        prog_bar = mmcv.ProgressBar(len(dataset)//world_size)
    loader_indices = data_loader.batch_sampler
    counter=0
    os.makedirs(out_dir,exist_ok=True)
    for batch_indices, data in zip(loader_indices, data_loader):
        if counter%world_size==rank:
            with torch.no_grad():
                img_tensor = data['img'][0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)
                crop_images, merge_fn = slide_inference(model, img_tensor.cuda())
                ts=[[] for i in range(len(crop_images))]
                zs=[[] for i in range(len(crop_images))]
                xs=[[] for i in range(len(crop_images))]
                outs=[]
                for i in range(len(crop_images)):
                    def call_back(log_z,log_x_recon,t,**args):
                        ts[i].append(t[0].item())
                        zs[i].append(resize(input=torch.exp(log_z[:,:-1]),size=crop_images[0].shape[2:],mode='bilinear',align_corners=model.align_corners).cpu())
                        xs[i].append(resize(input=torch.exp(log_x_recon[:,:-1]),size=crop_images[0].shape[2:],mode='bilinear',align_corners=model.align_corners).cpu())
                    out = model.sample(crop_images[i],return_logits = True,call_back=call_back)
                    out = out['logits'][:,:-1]
                    out = resize(input=out,size=crop_images[0].shape[2:],mode='bilinear',align_corners=model.align_corners)
                    outs.append(out)
                out=merge_fn(outs)
                zs=[merge_fn(i) for i in  zip(*zs)]
                xs=[merge_fn(i) for i in  zip(*xs)]
                ts=ts[0]
                img=imgs[0]
                img_meta=img_metas[0]
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]
                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))
                out=torch.argmax(resize(out, (ori_h, ori_w),mode='bilinear',align_corners=model.align_corners),1)
                zs=[torch.argmax(resize(i, (ori_h, ori_w),mode='bilinear',align_corners=model.align_corners),1) for i in zs]
                xs=[torch.argmax(resize(i, (ori_h, ori_w),mode='bilinear',align_corners=model.align_corners),1) for i in xs]
                torch.save(dict(out=out,zs=zs,ts=ts,xs=xs),"%s/%s_data.pkl"%(out_dir,counter))
                showed_img_out=model.show_result(img_show,out,opacity=opacity)
                showed_zt_out=make_grid([model.show_result(img_show,i,opacity=opacity) for i in zs])
                showed_xt_out=make_grid([model.show_result(img_show,i,opacity=opacity) for i in xs])
                cv2.imwrite("%s/%s_seg.png"%(out_dir,counter),showed_img_out)
                cv2.imwrite("%s/%s_zt.png"%(out_dir,counter),showed_zt_out)
                cv2.imwrite("%s/%s_xt.png"%(out_dir,counter),showed_xt_out)
                if rank==0:
                    prog_bar.update()
        counter+=1
        

def eval(config,out_dir):
    cfg = mmcv.Config.fromfile(config)
    dataset = build_dataset(cfg.data.test)
    import glob
    import json
    files=glob.glob("%s/*_data.pkl"%out_dir)
    files=sorted(files,key=lambda x:int(x.split('/')[-1][:-9]))
    xts=[torch.load(i)['xs'] for i in files]
    res=[]
    for T in range(len(xts[0])):
        g=[i[T].cpu().numpy().squeeze() for i in xts]
        miou = dataset.evaluate(g, metric="mIoU")
        res.append(miou)
    with open("%s/metrics.json"%out_dir,"w") as f:
        json.dump(res,f)
    targets=['mIoU','aAcc','mAcc']
    r2=[]
    for i in range(len(res)):
        g={j:res[-i][j] for j in targets}
        g["t"]=i
        r2.append(g)
    print(r2)
    with open("%s/metrics_simple.json"%out_dir,"w") as f:
        json.dump(r2,f)

        

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--out_dir")
    args=parser.parse_args()
    mp.set_start_method('spawn',force=True)
    world_size=torch.cuda.device_count()
    mp.spawn(worker,nprocs=world_size,args=(args.config,args.checkpoint,args.out_dir,world_size),daemon=False)
    eval(config,out_dir)
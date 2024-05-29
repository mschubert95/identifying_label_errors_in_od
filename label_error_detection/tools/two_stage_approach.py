# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import numpy as np
import os
import os.path as osp
import pandas as pd
import scipy
import time
import warnings

import mmcv
from mmdet import datasets
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.ops.nms import nms

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')

    args = parser.parse_args()

    return args


def preds_loss_assignment(out):
    boxes = np.array(out[1][0][0].detach().cpu().numpy())
    mask = boxes[:, -1] > 0.25
    ### PD baseline
    mask = boxes[:, -1] >= 0.0
    boxes = boxes[mask]
    inds = np.array(out[1][1][0].detach().cpu().numpy())[mask]
    res = np.array(out[1][2][0].detach().cpu().numpy())[mask]
    inds = np.array(out[1][1][0].detach().cpu().numpy())
    res = np.array(out[1][2][0].detach().cpu().numpy())

    loss_cls = [np.array(out[0]['loss_rpn_cls'][int(res[i])][int(inds[i])].detach().cpu().numpy()) for i, _ in enumerate(boxes)]

    loss_bbox = [np.array(out[0]['loss_rpn_bbox'][int(res[i])][int(inds[i])].detach().cpu().numpy()) for i, _ in enumerate(boxes)]

    if len(loss_bbox) > 0:
        loss_bbox = np.sum(loss_bbox, axis=1)
        boxes = np.column_stack((boxes, loss_cls, loss_bbox))
        return boxes
    else:
        return []


def dataframe(loss_list, img_paths, work_dir, save_name, n_classes=0):
    loss_list = torch.cat(loss_list, dim=0)
    if save_name == 'loss':
        df_gt = pd.DataFrame(loss_list.detach().cpu().numpy(), columns=[["xmin", "ymin", "xmax", "ymax", "s"] + [f"prob_{i}" for i in range(n_classes)] + ["prob_bg", "loss_cls", "loss_bbox", "class_id", "rpn_s", "rpn_cls_loss", "rpn_bbox_loss"]])
    df_gt["img_path"] = [x for xx in img_paths for x in xx]

    df_gt.to_csv(work_dir + "prediction_" + save_name + '.csv')


def detect_pred_loss(preds, labels, loss_tensor, stage_losses, inds, scores, sample_inds, rpn_loss, cfg, method='loss'):
    if method == 'loss':
        cls_loss = [loss_tensor['loss_bbox']['loss_cls'][x // len(cfg.data.test_pert.classes)].detach().cpu().numpy() for x in inds]
        bbox_loss = [torch.sum(loss_tensor['loss_bbox']['loss_bbox'][x // len(cfg.data.test_pert.classes)]).detach().cpu().numpy() for x in inds]

        nms_inds = [x.detach().cpu().numpy() // len(cfg.data.test_pert.classes) for x in inds]

        rpn_loss = rpn_loss[nms_inds]

    if scores.shape[1] != len(cfg.data.test_pert.classes)+1:
        scores = np.zeros((0,len(cfg.data.test_pert.classes)+1))


    if method == 'loss':
        preds = np.column_stack((preds, scores, cls_loss, bbox_loss, labels, rpn_loss[:, -3:]))
    elif method == 'score':
        preds = np.column_stack((preds, scores, labels))
    elif method == 'entropy':
        entr = scipy.stats.entropy(scipy.special.softmax(scores[:,:-1], axis=1), axis=1)
        preds = np.column_stack((preds, scores, entr, labels))

    return preds


def assignment(out_roi, preds_loss, cfg, method='loss'):
    rpn_inds = [x.detach().cpu().numpy() for x in out_roi[6]]
    for k in range(len(rpn_inds)):
        rpn_inds_k = rpn_inds[k]
        rpn_inds_k = rpn_inds_k[rpn_inds_k < preds_loss.shape[0]]
        preds_loss = preds_loss[rpn_inds_k]
        preds_loss = np.concatenate((preds_loss, np.zeros((len(rpn_inds[k])-len(rpn_inds_k), 7))), axis=0)

    preds = detect_pred_loss(np.array(out_roi[2].detach().cpu().numpy()), np.array(out_roi[3].detach().cpu().numpy()), out_roi[1], out_roi[0], out_roi[4], np.array(out_roi[5].detach().cpu().numpy()), out_roi[6], preds_loss, cfg, method=method)

    return preds


def prediction(model, data_loader, cfg):
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    img_preds_loss = []
    img_paths_loss = []
    for i, data in enumerate(data_loader):
        inputs, kwargs = model.scatter(data, dict(), [-1])
        inputs[0]['img'] = inputs[0]['img'].to("cuda:0")
        inputs[0]['gt_bboxes'][0] = inputs[0]['gt_bboxes'][0].to("cuda:0")
        inputs[0]['gt_labels'][0] = inputs[0]['gt_labels'][0].to("cuda:0")
        out = model.module.loss_heatmap_first_stage(**(inputs[0]))

        preds_loss = preds_loss_assignment(out)

        if len(preds_loss) > 0:
            inputs[0]['proposal_list'] = [torch.tensor(preds_loss[:, :5], device='cuda:0')]
        else:
            inputs[0]['proposal_list'] = [torch.tensor([], device='cuda:0')]
        
        # delete lines 162-171 for PD baseline and naive score
        gt_add = np.array(inputs[0]['gt_bboxes'][0].detach().cpu().numpy())
        zeros_add = np.zeros((len(inputs[0]['gt_bboxes'][0]), 3))
        zeros_add[:, 0] = 1 
        zeros_add[:, 1] = out[0]['loss_rpn_gt_cls'].detach().cpu().numpy()
        zeros_add[:, 2] = out[0]['loss_rpn_gt_bbox'].detach().cpu().numpy()
        add = np.concatenate((gt_add, zeros_add), axis=1)
        if len(preds_loss) > 0:
            preds_loss = np.concatenate((add, preds_loss), axis=0)
        else:
            preds_loss = add

        if len(preds_loss) > 0:
            ### Loss
            inputs[0]['first_stage_loss'] = torch.tensor(preds_loss[:,-2:], device='cuda:0')
            print(inputs[0]['first_stage_loss'])
            inputs[0]['first_stage_loss'] = None

            out_roi_loss = model.module.loss_heatmap_second_stage(**(inputs[0]))

            if len(out_roi_loss[2].detach().cpu().numpy()) > 0:
                ts_preds_loss = assignment(out_roi_loss, preds_loss, cfg, method='loss')

                img_preds_loss.append(torch.tensor(ts_preds_loss, device = 'cuda:0'))
                img_paths_loss.append([inputs[0]['img_metas'][0]['filename']]*ts_preds_loss.shape[0])

        prog_bar.update()
            
    dataframe(img_preds_loss, img_paths_loss, cfg.work_dir, 'loss', len(cfg.data.test_pert.classes))
    cfg.model.train_cfg.rcnn[0]["sampler"]["add_gt_as_proposals"] = True
    cfg.model.train_cfg.rcnn[1]["sampler"]["add_gt_as_proposals"] = True
    cfg.model.train_cfg.rcnn[2]["sampler"]["add_gt_as_proposals"] = True   

def hyperparameter_setting(cfg, string, score_thr=None):

    if string == "prediction":
        cfg.model.train_cfg.rpn.assigner['pos_iou_thr']=0.7
        cfg.model.train_cfg.rpn.assigner['neg_iou_thr']=0.3
        cfg.model.train_cfg.rpn.assigner['min_pos_iou']=0.3
        cfg.model.train_cfg.rpn.sampler['type']='PseudoSampler'
        cfg.model.train_cfg.rpn_proposal["nms_pre"]=500000
        cfg.model.train_cfg.rpn_proposal["max_per_img"]=500000
        cfg.model.train_cfg.rpn_proposal.nms["iou_threshold"]=0.5
        if score_thr is not None:
            cfg.model.test_cfg.rcnn["score_thr"] = score_thr
        else:
            cfg.model.test_cfg.rcnn["score_thr"] = 0.0
        cfg.model.test_cfg.rcnn.nms["iou_threshold"]=0.5
        cfg.model.test_cfg.rcnn["max_per_img"]=10000000000

        if 'metrics' in cfg.model.test_cfg.rcnn:
            cfg.model.test_cfg.rcnn['metrics'] = True
        if 'metrics' in cfg.model.test_cfg.rpn:
            cfg.model.test_cfg.rpn['metrics'] = True

    return cfg


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir

    if not osp.exists(cfg.work_dir): os.makedirs(cfg.work_dir)




    hyperparameter_setting(cfg, 'prediction')

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = datasets.CLASSES    

    cfg.data.test_pert.test_mode = False

    test_pert_dataset = build_dataset(cfg.data.test_pert)
    test_pert_dataloader = build_dataloader(test_pert_dataset,
                                        samples_per_gpu=1,
                                        workers_per_gpu=5,
                                        dist=False,
                                        shuffle=False)

    model = MMDataParallel(model, device_ids=[0])
    model.train()

    prediction(model, test_pert_dataloader, cfg)


if __name__ == '__main__':
    main()

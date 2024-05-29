import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from mmcv import Config, DictAction

from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'work_dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--save-dir',
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


def load_dataframes(work_dir):
    if osp.exists(work_dir + 'prediction_loss.csv'):
        df_loss = pd.read_csv(work_dir + 'prediction_loss.csv').drop('Unnamed: 0', axis=1)
    
    
    return df_loss


def load_annotations_in_df(path, cfg, shifts=False):
    data = json.load(open(path))

    images = [x['file_name'] for x in data['images']]
    ids = [x['id'] for x in data['images']]
    image_ids = [images[x['image_id']] for x in data['annotations'] if x['image_id'] in ids]
    bboxes = [list(map(int, x['bbox'])) for x in data['annotations'] if x['image_id'] in ids]
    category_ids = [x['category_id'] for x in data['annotations'] if x['image_id'] in ids]
    box_ids = [x['id'] for x in data['annotations'] if x['image_id'] in ids]
    if shifts:
        shift_candidates = [str(int(x['id'])) for x in data['annotations'] if x['image_id'] in ids and x['shift']==True]

    df = pd.DataFrame(np.concatenate((np.asmatrix(bboxes), np.asmatrix(category_ids).T, np.asmatrix(image_ids).T, np.asmatrix(box_ids).T), axis=1), columns=['xmin', 'ymin', 'xmax', 'ymax', 'class_id', 'img_path', 'box_id'])
    df[["xmin", "ymin", "xmax", "ymax", "class_id"]] = df[["xmin", "ymin", "xmax", "ymax", "class_id"]].astype(float)
    
    df['xmax'] += df['xmin']
    df['ymax'] += df['ymin']

    img_scale = [x["img_scale"] for x in cfg.data.test_pert.pipeline if "img_scale" in x.keys()][0]
    x = img_scale[0]
    y = img_scale[1]

    x_scale =  x / data['images'][0]['width']
    y_scale = y / data['images'][0]['height']

    df['xmin'] *= x_scale
    df['xmax'] *= x_scale
    df['ymin'] *= y_scale
    df['ymax'] *= y_scale

    df["class_id"] -= 1

    if shifts:
        return df, shift_candidates
    else:
        return df


def error_candidates(df_clean, df_le):
    clean_ids = list(df_clean["box_id"])
    clean_classes = list(df_clean["class_id"])
    pert_drop_ids = list(df_le["box_id"])
    pert_flip_ids = list(df_le["box_id"])
    pert_flip_classes = list(df_le["class_id"])
    pert_spawn_ids = list(df_le["box_id"])

    spawn_candidates = [x for x in pert_spawn_ids if x not in clean_ids]

    drop_candidates = [x for x in clean_ids if x not in pert_drop_ids]

    flip_candidates = [x for i,x in enumerate(clean_ids) if x in pert_flip_ids and clean_classes[i]!=pert_flip_classes[pert_flip_ids.index(x)]]

    return spawn_candidates, drop_candidates, flip_candidates


def pert_eval(df, df_pert, df_clean, spawn_candidates, drop_candidates, flip_candidates, shift_candidates, save_dir, img_prefix, n_classes, iou_thr=0.2, method='loss'):
    if method == 'loss':
        df['loss'] = df['loss_cls'] + df['loss_bbox'] + df['rpn_cls_loss'] + df['rpn_bbox_loss']
        df = df.sort_values(by=['loss'], ascending=False).reset_index(drop=True)

    df_clean["used"] = 0
    df_pert["used"] = 0
    df["label_error"] = 0
    df["spawn"] = 0
    df["drop"] = 0
    df["flip"] = 0
    df["shift"] = 0
    df["fp"] = 1
    df["tp"] = 0
    df["gt_bbox_id"] = -1

    paths = list(set(df["img_path"]))

    print("Pred detection...")
    for i in tqdm(range(len(paths))):
        df_clean = df_clean.loc[df_clean["used"]==0].reset_index(drop=True)
        df_pert = df_pert.loc[df_pert["used"]==0].reset_index(drop=True)

        df_pert_img = df_pert.loc[df_pert["img_path"] == paths[i].split("/")[-1]]
        df_img = df.loc[df["img_path"] == paths[i]]
        if method == 'loss':
            df_img = df_img.sort_values(by=['loss'], ascending=False)

        pred = torch.tensor(df_img.loc[:, ["xmin", "ymin", "xmax", "ymax"]].to_numpy().astype(float))
        gt = torch.tensor(df_pert_img.loc[:, ["xmin", "ymin", "xmax", "ymax"]].to_numpy().astype(float))

        ious = bbox_overlaps(pred, gt).detach().cpu().numpy()
        ious_class = np.stack([np.where(df_pert_img.loc[:, "class_id"]==df_img.loc[j, "class_id"], ious[j_id], 0) for j_id, j in enumerate(list(df_img.index))])

        df_clean_img = df_clean.loc[df_clean["img_path"] == paths[i].split("/")[-1]]
        gt_clean = torch.tensor(df_clean_img.loc[:, ["xmin", "ymin", "xmax", "ymax"]].to_numpy().astype(float))
        ious_clean = bbox_overlaps(pred, gt_clean).detach().cpu().numpy()
        ious_clean_class = np.stack([np.where(df_clean_img.loc[:, "class_id"]==df_img.loc[j, "class_id"], ious_clean[j_id], 0) for j_id, j in enumerate(list(df_img.index))])
        
        if ious.shape[1] == 0:
            ious = np.zeros((len(df_img), 1))
            ious_class = np.zeros((len(df_img), 1))
        if ious_clean.shape[1] == 0:
            ious_clean = np.zeros((len(df_img), 1))
            ious_clean_class = np.zeros((len(df_img), 1))

        for i_id, ind in enumerate(list(df_img.index)):
            max_iou = np.argmax(ious[i_id])
            max_iou_class = np.argmax(ious_class[i_id])
            max_iou_clean = np.argmax(ious_clean[i_id])
            max_iou_clean_class = np.argmax(ious_clean_class[i_id])
            if np.max(ious[i_id:]) < iou_thr and np.max(ious_clean[i_id:]) < iou_thr:
                break
            elif np.max(ious[i_id]) >= iou_thr or np.max(ious_clean[i_id]) >= iou_thr:
                if df_pert_img.shape[0] > 0:
                    box_id = df_pert.loc[df_pert_img.index[max_iou], "box_id"]
                    box_id_class = df_pert.loc[df_pert_img.index[max_iou_class], "box_id"]
                else:
                    box_id = -1
                    box_id_class = -1
                if df_clean_img.shape[0] > 0:
                    box_id_clean = df_clean.loc[df_clean_img.index[max_iou_clean], "box_id"]
                    box_id_clean_class = df_clean.loc[df_clean_img.index[max_iou_clean_class], "box_id"]
                else:
                    box_id_clean = -1
                    box_id_clean_class = -1
                if np.max(ious[i_id]) == np.max(ious_clean[i_id]):
                    if str(int(box_id)) in flip_candidates: # Flip
                        df.loc[ind, "label_error"] = 1
                        df.loc[ind, "flip"] = 1
                        df.loc[ind, "fp"] = 0
                        df_clean.loc[df_clean_img.index[max_iou_clean_class], "used"] = 1
                        df_pert.loc[df_pert["box_id"]==box_id, "used"] = 1
                        ious[:, list(df_pert_img.index).index(df_pert.loc[df_pert["box_id"]==box_id].index)] = 0
                        ious_class[:, list(df_pert_img.index).index(df_pert.loc[df_pert["box_id"]==box_id].index)] = 0
                        ious_clean[:, max_iou_clean] = 0
                        ious_clean_class[:, max_iou_clean] = 0
                        df.loc[ind, "gt_bbox_id"] = box_id
                    else:
                        if str(int(box_id_class)) in flip_candidates:
                            if np.max(ious_clean_class[i_id]) >= iou_thr: # Flip
                                df.loc[ind, "label_error"] = 1
                                df.loc[ind, "flip"] = 1
                                df.loc[ind, "fp"] = 0
                                df_clean.loc[df_clean_img.index[max_iou_clean_class], "used"] = 1
                                df_pert.loc[df_pert["box_id"]==box_id, "used"] = 1
                                ious[:, list(df_pert_img.index).index(df_pert.loc[df_pert["box_id"]==box_id].index)] = 0
                                ious_class[:, list(df_pert_img.index).index(df_pert.loc[df_pert["box_id"]==box_id].index)] = 0
                                ious_clean[:, max_iou_clean] = 0
                                ious_clean_class[:, max_iou_clean] = 0
                                df.loc[ind, "gt_bbox_id"] = box_id
                        else: 
                            if np.max(ious_clean[i_id]) >= iou_thr: # TP or Spawn?
                                assert box_id == box_id_clean
                                iou_max = np.max(ious[i_id])
                                iou_clean_max = np.max(ious_clean[i_id])
                                second_iou_max = 0
                                second_iou_clean_max = 0
                                if len(ious[i_id]) > 1:
                                    second_iou_max = np.max(ious[i_id][ious[i_id] < iou_max])
                                if len(ious_clean[i_id]) > 1:
                                    second_iou_clean_max = np.max(ious_clean[i_id][ious_clean[i_id] < iou_clean_max])
                                
                                if second_iou_max >= iou_thr and second_iou_max > second_iou_clean_max and str(int(df_pert.loc[df_pert_img.index[np.where(ious[i_id] == second_iou_max)], "box_id"])) in spawn_candidates: # Spawn
                                    df.loc[ind, "fp"] = 0
                                    df.loc[ind, "label_error"] = 1
                                    if iou_clean_max >= iou_thr:
                                        df.loc[ind, "spawn"] = 2
                                    else:
                                        df.loc[ind, "spawn"] = 1
                                    df_pert.loc[df_pert_img.index[np.where(ious[i_id] == second_iou_max)], "used"] = 1
                                    df.loc[ind, "gt_bbox_id"] = str(int(df_pert.loc[df_pert_img.index[np.where(ious[i_id] == second_iou_max)], "box_id"]))
                                    ious_class[:, np.where(ious[i_id] == second_iou_max)] = 0
                                    ious[:, np.where(ious[i_id] == second_iou_max)] = 0
                                else: # TP
                                    if np.max(ious_clean_class[i_id]) >= iou_thr:
                                        if box_id_class == box_id_clean_class:
                                            df.loc[ind, "tp"] = 1   
                                            df.loc[ind, "fp"] = 0
                                            df_pert.loc[df_pert["box_id"]==box_id_class, "used"] = 1
                                            df_clean.loc[df_clean["box_id"]==box_id_clean_class, "used"] = 1
                                            ious_clean[:, max_iou_clean_class] = 0
                                            ious_clean_class[:, max_iou_clean_class] = 0
                                            ious[:, max_iou_class] = 0
                                            ious_class[:, max_iou_class] = 0
                                            df.loc[ind, "gt_bbox_id"] = box_id_class
                                        elif str(int(box_id_clean_class)) in drop_candidates: # Drop
                                            df.loc[ind, "fp"] = 0
                                            df.loc[ind, "label_error"] = 1
                                            ious_clean[:, max_iou_clean_class] = 0
                                            ious_clean_class[:, max_iou_clean_class] = 0
                                            df.loc[ind, "drop"] = 1
                                            df_clean.loc[df_clean["box_id"]==box_id_clean_class, "used"] = 1
                                            df.loc[ind, "gt_bbox_id"] = box_id_clean_class
                                        elif str(int(box_id_clean_class)) in flip_candidates: # Flip
                                            df.loc[ind, "label_error"] = 1
                                            df.loc[ind, "flip"] = 1
                                            df.loc[ind, "fp"] = 0
                                            df_clean.loc[df_clean_img.index[max_iou_clean_class], "used"] = 1
                                            df_pert.loc[df_pert["box_id"]==box_id_clean_class, "used"] = 1
                                            ious[:, list(df_pert_img.index).index(df_pert.loc[df_pert["box_id"]==box_id_clean_class].index)] = 0
                                            ious_class[:, list(df_pert_img.index).index(df_pert.loc[df_pert["box_id"]==box_id_clean_class].index)] = 0
                                            ious_clean[:, max_iou_clean_class] = 0
                                            ious_clean_class[:, max_iou_clean_class] = 0
                                            df.loc[ind, "gt_bbox_id"] = box_id_clean_class
                                        elif str(int(box_id_clean_class)) in shift_candidates: # Shift
                                            df.loc[ind, "fp"] = 0
                                            df.loc[ind, "label_error"] = 1
                                            ious_clean[:, max_iou_clean_class] = 0
                                            ious_clean_class[:, max_iou_clean_class] = 0
                                            df_pert.loc[df_pert["box_id"]==box_id_clean_class, "used"] = 1
                                            df.loc[ind, "shift"] = 1
                                            df_clean.loc[df_clean["box_id"]==box_id_clean_class, "used"] = 1    
                                            ious[:, list(df_pert_img.index).index(df_pert.loc[df_pert["box_id"]==box_id_clean_class].index)] = 0
                                            ious_class[:, list(df_pert_img.index).index(df_pert.loc[df_pert["box_id"]==box_id_clean_class].index)] = 0
                                            df.loc[ind, "gt_bbox_id"] = box_id_clean_class
                                        else:
                                            print(WRONG)
                elif np.max(ious[i_id]) > np.max(ious_clean[i_id]):
                    assert str(int(box_id)) in spawn_candidates or str(int(box_id)) in shift_candidates
                    df.loc[ind, "fp"] = 0
                    df.loc[ind, "label_error"] = 1
                    ious[:, max_iou] = 0
                    ious_class[:, max_iou] = 0
                    df_pert.loc[df_pert["box_id"]==box_id, "used"] = 1
                    if str(int(box_id)) in spawn_candidates: # Spawn
                        if np.max(ious_clean[i_id]) >= iou_thr:
                            df.loc[ind, "spawn"] = 2
                        else:
                            df.loc[ind, "spawn"] = 1
                        df.loc[ind, "gt_bbox_id"] = box_id
                    elif str(int(box_id)) in shift_candidates: # Shift
                        df.loc[ind, "shift"] = 1
                        df_clean.loc[df_clean["box_id"]==box_id, "used"] = 1    
                        ious_clean[:, list(df_clean_img.index).index(df_clean.loc[df_clean["box_id"]==box_id].index)] = 0
                        ious_clean_class[:, list(df_clean_img.index).index(df_clean.loc[df_clean["box_id"]==box_id].index)] = 0
                        df.loc[ind, "gt_bbox_id"] = box_id
                elif np.max(ious[i_id]) < np.max(ious_clean[i_id]):
                    assert str(int(box_id_clean)) in drop_candidates or str(int(box_id_clean)) in shift_candidates
                    df.loc[ind, "fp"] = 0
                    df.loc[ind, "label_error"] = 1
                    ious_clean[:, max_iou_clean] = 0
                    ious_clean_class[:, max_iou_clean] = 0
                    df_clean.loc[df_clean["box_id"]==box_id_clean, "used"] = 1
                    if str(int(box_id_clean)) in drop_candidates: # Drop
                        df.loc[ind, "drop"] = 1
                        df.loc[ind, "gt_bbox_id"] = box_id_clean
                    elif str(int(box_id_clean)) in shift_candidates: # Shift
                        df.loc[ind, "shift"] = 1
                        df_pert.loc[df_pert["box_id"]==box_id_clean, "used"] = 1    
                        ious[:, list(df_pert_img.index).index(df_pert.loc[df_pert["box_id"]==box_id_clean].index)] = 0
                        ious_class[:, list(df_pert_img.index).index(df_pert.loc[df_pert["box_id"]==box_id_clean].index)] = 0
                        df.loc[ind, "gt_bbox_id"] = box_id_clean

    df_clean_not_used = df_clean.loc[df_clean["used"]==0].reset_index(drop=True)
    df_clean_not_used = df_clean_not_used.loc[df_clean_not_used["box_id"].isin(drop_candidates)].reset_index(drop=True)
    df_clean_not_used["s"] = -1
    if method == 'loss':
        df_clean_not_used["loss_cls"] = -1
        df_clean_not_used["loss_bbox"] = -1
        df_clean_not_used["rpn_s"] = -1
        df_clean_not_used["rpn_cls_loss"] = -1
        df_clean_not_used["rpn_bbox_loss"] = -1
        df_clean_not_used["loss"] = -1
   
    df_clean_not_used["label_error"] = 1
    df_clean_not_used["spawn"] = 0
    df_clean_not_used["drop"] = 1
    df_clean_not_used["flip"] = 0
    df_clean_not_used["shift"] = 0
    df_clean_not_used["fp"] = 0
    df_clean_not_used["tp"] = 0
    df_clean_not_used["img_path"] = img_prefix + df_clean_not_used["img_path"]

    for i in range(n_classes):
        df_clean_not_used["prob_"+str(i)] = -1
    df_clean_not_used["prob_bg"] = -1

    df_spawn_not_used = df_pert.loc[df_pert["used"]==0].reset_index(drop=True)
    df_spawn_not_used = df_spawn_not_used.loc[df_spawn_not_used["box_id"].isin(spawn_candidates)].reset_index(drop=True)
    df_spawn_not_used["s"] = -1
    if method == 'loss':
        df_spawn_not_used["loss_cls"] = -1
        df_spawn_not_used["loss_bbox"] = -1
        df_spawn_not_used["rpn_s"] = -1
        df_spawn_not_used["rpn_cls_loss"] = -1
        df_spawn_not_used["rpn_bbox_loss"] = -1
        df_spawn_not_used["loss"] = -1
    
    df_spawn_not_used["label_error"] = 1
    df_spawn_not_used["spawn"] = 1
    df_spawn_not_used["drop"] = 0
    df_spawn_not_used["flip"] = 0
    df_spawn_not_used["shift"] = 0
    df_spawn_not_used["fp"] = 0
    df_spawn_not_used["tp"] = 0
    df_spawn_not_used["img_path"] = img_prefix + df_spawn_not_used["img_path"]

    for i in range(n_classes):
        df_spawn_not_used["prob_"+str(i)] = -1
    df_spawn_not_used["prob_bg"] = -1

    df_flip_not_used = df_pert.loc[df_pert["used"]==0].reset_index(drop=True)
    df_flip_not_used = df_flip_not_used.loc[df_flip_not_used["box_id"].isin(flip_candidates)].reset_index(drop=True)
    df_flip_not_used["s"] = -1
    if method == 'loss':
        df_flip_not_used["loss_cls"] = -1
        df_flip_not_used["loss_bbox"] = -1
        df_flip_not_used["rpn_s"] = -1
        df_flip_not_used["rpn_cls_loss"] = -1
        df_flip_not_used["rpn_bbox_loss"] = -1
        df_flip_not_used["loss"] = -1
    
    df_flip_not_used["label_error"] = 1
    df_flip_not_used["spawn"] = 0
    df_flip_not_used["drop"] = 0
    df_flip_not_used["flip"] = 1
    df_flip_not_used["shift"] = 0
    df_flip_not_used["fp"] = 0
    df_flip_not_used["tp"] = 0
    df_flip_not_used["img_path"] = img_prefix + df_flip_not_used["img_path"]

    for i in range(n_classes):
        df_flip_not_used["prob_"+str(i)] = -1
    df_flip_not_used["prob_bg"] = -1

    df_shift_not_used = df_pert.loc[df_pert["used"]==0].reset_index(drop=True)
    df_shift_not_used = df_shift_not_used.loc[df_shift_not_used["box_id"].isin(shift_candidates)].reset_index(drop=True)
    df_shift_not_used["s"] = -1
    if method == 'loss':
        df_shift_not_used["loss_cls"] = -1
        df_shift_not_used["loss_bbox"] = -1
        df_shift_not_used["rpn_s"] = -1
        df_shift_not_used["rpn_cls_loss"] = -1
        df_shift_not_used["rpn_bbox_loss"] = -1
        df_shift_not_used["loss"] = -1
    
    df_shift_not_used["label_error"] = 1
    df_shift_not_used["spawn"] = 0
    df_shift_not_used["drop"] = 0
    df_shift_not_used["flip"] = 0
    df_shift_not_used["shift"] = 1
    df_shift_not_used["fp"] = 0
    df_shift_not_used["tp"] = 0
    df_shift_not_used["img_path"] = img_prefix + df_shift_not_used["img_path"]

    for i in range(n_classes):
        df_shift_not_used["prob_"+str(i)] = -1
    df_shift_not_used["prob_bg"] = -1

    if method == 'loss':
        columns = ["xmin", "ymin", "xmax", "ymax", "s"] + [f"prob_{i}" for i in range(n_classes)] + ["prob_bg", "loss_cls", "loss_bbox", "class_id", "rpn_s", "rpn_cls_loss", "rpn_bbox_loss", "img_path", "loss", "label_error", "spawn", "drop", "flip", "shift", "fp", "tp"]
        
    df = pd.concat([df, df_clean_not_used.loc[:, columns], df_spawn_not_used.loc[:, columns], df_flip_not_used.loc[:, columns], df_shift_not_used.loc[:, columns]], axis=0, ignore_index=True).reset_index(drop=True)

    df.to_csv(save_dir + "pert_assignment_" + method + ".csv")

    print("Done!")



def main():

    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark

    cfg.work_dir = args.work_dir + '/'
    if not args.save_dir:
        cfg.save_dir = cfg.work_dir
    else:
        cfg.save_dir = args.save_dir + '/'

    df_loss = load_dataframes(cfg.work_dir)

    df_clean = load_annotations_in_df(cfg.data.test.ann_file, cfg)
    df_le, shift_candidates = load_annotations_in_df(cfg.data.test_pert.ann_file, cfg, shifts=True)

    spawn_candidates, drop_candidates, flip_candidates = error_candidates(df_clean, df_le)

    pert_eval(df_loss, df_le, df_clean, spawn_candidates, drop_candidates, flip_candidates, shift_candidates, cfg.save_dir, cfg.data.test.img_prefix, len(cfg.data.test_pert.classes), method = 'loss')


if __name__ == '__main__':
    main()

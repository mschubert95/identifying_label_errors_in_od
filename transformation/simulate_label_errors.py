import matplotlib.image as mpimg
import os
import json
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

import torch
import torchvision.ops.boxes as bops


def iou2d(orig, new):
    orig[2] += orig[0]
    orig[3] += orig[1]
    new[2] += new[0]
    new[3] += new[1]

    box1 = torch.tensor([orig], dtype=torch.float)
    box2 = torch.tensor([new], dtype=torch.float)
    iou = bops.box_iou(box1, box2)[0][0].detach().cpu().numpy()

    return iou

 
def main(path, rate=0.05, save_path=None, n_classes=26):
    
    images = []
    ids = []
    image_ids = []
    bboxes = []
    category_ids = []

    data = json.load(open(path, "r"))
    
    rate = rate / 4
    
    images = [x['file_name'] for x in data['images']]
    data["images"] = [x for x in data["images"] if x["file_name"] in images]
    print(len(data["images"]))
    image_sizes = data["images"][0]["width"], data["images"][0]["height"]
    ids = [x['id'] for x in data['images']]
    image_ids = [x['image_id'] for x in data['annotations']]
    bboxes = [x['bbox'] for x in data['annotations']]
    category_ids = [x['category_id'] for x in data['annotations']]
    box_ids = [x['id'] for x in data['annotations']]

    df = pd.DataFrame(np.concatenate((np.asmatrix(bboxes), np.asmatrix(category_ids).T, np.asmatrix(image_ids).T, np.asmatrix(box_ids).T), axis=1), columns=['x', 'y', 'w', 'h', 'c', 'i_id', 'box_id'])
    
    df = df.drop_duplicates(keep=False).reset_index(drop=True)
    
    n_rate = int(len(df)*rate)
    
    ### Drops
    df_kick = df.sample(n=int(n_rate))

    df_kick = df_kick.reset_index(drop=True)

    df = pd.concat([df, df_kick]).drop_duplicates(keep=False).reset_index(drop=True)
    
    ### Flips
    df['c_pert'] = df['c']
    categories = np.arange(1,n_classes+1)
    df = df.sample(frac=1).reset_index(drop=True)
    cat_pre = list(df.loc[:n_rate, 'c'])
    cat_post = random.choices(categories, k=len(cat_pre))
    for i in range(len(cat_pre)):
        while cat_pre[i] == cat_post[i]:
            cat_post[i] = random.choices(categories, k=1)[0]
    df.loc[:n_rate, 'c_pert'] = cat_post

    ### Shifts
    shifts = list(df.loc[n_rate:2*n_rate, "box_id"])
    for k in range(len(df.loc[n_rate:2*n_rate])):
        iou = 0
        while iou < 0.4 or iou >= 0.7:
            w_new = np.random.normal(df.loc[n_rate+k, "w"], 0.15*df.loc[n_rate+k, "w"])
            h_new = np.random.normal(df.loc[n_rate+k, "h"], 0.15*df.loc[n_rate+k, "h"])
            x_new = np.clip(np.random.normal(df.loc[n_rate+k, "x"]+df.loc[n_rate+k, "w"]/2, 0.15*df.loc[n_rate+k, "w"])-df.loc[n_rate+k, "w"]/2, 0, image_sizes[0])
            y_new = np.clip(np.random.normal(df.loc[n_rate+k, "y"]+df.loc[n_rate+k, "h"]/2, 0.15*df.loc[n_rate+k, "h"])-df.loc[n_rate+k, "h"]/2, 0, image_sizes[1])
            iou = iou2d(list(df.loc[n_rate+k, ["x", "y", "w", "h"]]), [x_new, y_new, w_new, h_new])

        df.loc[n_rate+k, "x"] = x_new
        df.loc[n_rate+k, "y"] = y_new
        df.loc[n_rate+k, "w"] = w_new
        df.loc[n_rate+k, "h"] = h_new
    
    
    ### Spawns
    image_paths = list(set(df['i_id']))
    df_add = df.sample(frac=1).reset_index(drop=True)
    df_add = df_add.loc[:n_rate, :].reset_index(drop=True)
    
    img_ids_pre = list(df_add['i_id'])
    img_ids_post = random.choices(image_paths, k=len(df_add))
    for i in range(len(img_ids_pre)):
        while img_ids_pre[i] == img_ids_post[i]:
            img_ids_post[i] = random.choices(image_paths, k=1)[0]
    
    df_add['i_id'] = img_ids_post
    df_add['box_id'] = np.arange(len(box_ids), len(box_ids)+len(df_add))
    df_add['c_pert'] = df_add['c']
    df = pd.concat([df, df_add]).reset_index(drop=True)

    df['c'] = df['c_pert']

    df = df.sample(frac=1).reset_index(drop=True)

    anns = []
    
    for i in tqdm(range(len(images))):
        df_image = df.loc[df['i_id']==ids[i]].reset_index(drop=True)
        df_image = df_image.sort_values(by=['box_id']).reset_index(drop=True)

        for j in range(len(df_image)):
            ann_dict = {
                "segmentation": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                "area": int(df_image.loc[j, 'size']),
                "iscrowd": 0,
                "image_id": int(ids[i]),
                "bbox": [int(df_image.loc[j, 'x']), int(df_image.loc[j, 'y']), int(df_image.loc[j, 'w']), int(df_image.loc[j, 'h'])],
                "category_id": int(df_image.loc[j, 'c']),
                "id": int(df_image.loc[j, 'box_id']),
                "shift": True if int(df_image.loc[j, 'box_id']) in shifts else False
            }
            anns.append(ann_dict)


    coco_fmt_dict = {
    "info": data['info'],
    "licenses": data['licenses'],
    "images": data['images'],
    "annotations": anns,
    "categories": data['categories']
    }

    if save_path:
        json.dump(coco_fmt_dict, open(save_path, "w"))
 
 
if __name__ == '__main__':
    n_classes = int("number_of_classes")
    path = 'ann_path_without_label_errors.json'
    rate=0.2
    save_path = 'save_path.json'
    main(path, rate, save_path, n_classes)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json, shutil, random
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

def yolo_bbox(size, bbox):
    W, H = size
    x, y, w, h = bbox
    cx = (x + w / 2) / W
    cy = (y + h / 2) / H
    return cx, cy, w / W, h / H

def main(csv_path, img_dir, out_dir, folds=5, seed=42):
    df = pd.read_csv(csv_path)
    grp = df.groupby("image_id")

    for split in ["train", "val"]:
        os.makedirs(f"{out_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{out_dir}/labels/{split}", exist_ok=True)

    ids = sorted(grp.groups.keys())
    # use source as stratification
    strat = df.groupby("image_id")["source"].first().reindex(ids).fillna(0)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    # only keep the first fold
    _, val_idx = next(skf.split(ids, strat))
    val_set = set([ids[i] for i in val_idx])

    for img_id, g in tqdm(grp, desc="轉換標註"):
        split = "val" if img_id in val_set else "train"
        img_file = f"{img_id}.jpg"
        shutil.copyfile(f"{img_dir}/{img_file}", f"{out_dir}/images/{split}/{img_file}")

        with open(f"{out_dir}/labels/{split}/{img_id}.txt", "w") as f:
            for _, row in g.iterrows():
                bbox = json.loads(row["bbox"])
                w, h = row["width"], row["height"]
                cx, cy, bw, bh = yolo_bbox((w, h), bbox)
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        

    # 產生 YAML
    yaml_dict = {
        "train": f"{out_dir}/images/train",
        "val":   f"{out_dir}/images/val",
        "nc": 1,
        "names": ["wheat_head"]
    }

    import yaml as pyaml

    with open(f"./wheat.yaml", "w") as f:
        pyaml.dump(yaml_dict, f)
    print("✅ wheat.yaml created")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="gwhd_2020/train.csv")
    p.add_argument("--img_dir", default="gwhd_2020/train")
    p.add_argument("--out_dir", default="datasets/2020")
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args()
    main(args.csv, args.img_dir, args.out_dir, args.csv2, args.img_dir2, args.folds)

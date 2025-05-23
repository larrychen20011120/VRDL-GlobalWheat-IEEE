#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8-X 1024×1024 微調小麥偵測
"""
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG

print(DEFAULT_CFG)

if __name__ == "__main__":
    model = YOLO("yolov8x.pt")  # 大模型，亦可改 yolov8l.pt

    results = model.train(
        data="wheat_2021.yaml",
        epochs=100,
        imgsz=1024,
        batch=8,           # RTX 3090 24GB 實測可行，如不足可降為 4 並啟用 accumulate
        device=0,
        optimizer="AdamW",
        lr0=5e-4,
        cos_lr=True,
        warmup_epochs=5,
        weight_decay=1e-4,
        close_mosaic=10,   # 最後 10 個 epoch 關掉 Mosaic / MixUp
        patience=20,       # 早停
        workers=8,         # DataLoader threads
        #### Add Augmentation ###
        hsv_h=0.2,
        hsv_s=0.2,
        hsv_v=0.2,
        flipud=0.5,
        fliplr=0.5,
        mixup=0.2,
        cutmix=0.2,
        copy_paste=0.2,
        #########################
        project="runs",
        name="yolov8_wheat_2021"
    )

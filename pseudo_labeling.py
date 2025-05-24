import os, glob, torch, numpy as np, pandas as pd
import argparse, os, json, shutil, random
from ultralytics import YOLO
from tqdm.auto import tqdm
from prepare_dataset import yolo_bbox

def makePseudolabel(model, dest, CONF_TH=0.5, IOU_TH=0.6, TTA=True):
    # Inference for Global Wheat Detection – fixed bbox + tuneable threshold
    
    IMGSZ   = 1024
    BATCH   = 8

    # 03 inference
    TEST_DIR = "GlobalWheatTestSet"
            
    test_imgs = sorted(glob.glob(f"{TEST_DIR}/**/*.png"))
    
    for i in tqdm(range(0, len(test_imgs), BATCH)):
        paths = test_imgs[i:i+BATCH]
        preds = model.predict(
            source=paths,
            imgsz=IMGSZ,
            conf=CONF_TH,
            iou=IOU_TH,
            half=False,
            device=device,
            verbose=False,
            augment=TTA,
        )
        
        for res in preds:
            img_id = os.path.basename(res.path).split('.')[0]
            xyxy   = res.boxes.xyxy.cpu().numpy()      # (x1, y1, x2, y2)
            scores = res.boxes.conf.cpu().numpy()
    
            if len(xyxy):
                h_img, w_img = res.orig_shape
                xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w_img - 1)
                xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h_img - 1)
                w_h   = xyxy[:, 2:] - xyxy[:, :2]       # w = x2-x1, h = y2-y1
                xywh  = np.hstack([xyxy[:, :2], w_h])   # (x, y, w, h)

                img_file = f"{img_id}.png"
                shutil.copyfile(
                    res.path, 
                    f"{dest}/images/train/{img_file}"
                )

                with open(f"{dest}/labels/train/{img_id}.txt", "w") as f:
                
                    for (x, y, w, h), s in zip(xywh, scores):

                        if w > 0 and h > 0:
                            cx, cy, bw, bh = yolo_bbox((w_img, h_img), (x, y, w, h))
                            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


if __name__ == "__main__":

    # load the best model
    WEIGHT_FILE = "runs/yolov8_wheat_2021/weights/best.pt"
    SRC = "datasets/2021"
    DEST = "datasets/2021_pl"

    # copy the orinal dataset to pseudo labeling dataset
    os.system(f"cp -r {SRC} {DEST}")
    
    model    = YOLO(WEIGHT_FILE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.fuse()
    model.to(device).eval()
    
    # # generate pseudo labels
    makePseudolabel(model, DEST, CONF_TH=0.5, IOU_TH=0.5, TTA=True)

    # # 檢查有多少資料
    print("train size / origin:", len(os.listdir(f"{SRC}/images/train")))
    print("valid size / origin:", len(os.listdir(f"{SRC}/images/val")))
    print("train size / withpl:", len(os.listdir(f"{DEST}/images/train")))
    print("valid size / withpl:", len(os.listdir(f"{DEST}/images/val")))
    
    EPOCHS = 10
    # continual training
    results = model.train(
        data="pl.yaml",
        epochs=EPOCHS,
        imgsz=1024,
        batch=8,           
        device=0,
        optimizer="AdamW",
        lr0=5e-4,
        cos_lr=True,
        warmup_epochs=2,
        weight_decay=1e-4,
        close_mosaic=EPOCHS,   # Pseodo Labeling does not need mosaic
        patience=20,       
        workers=8,         
        #### Add Augmentation ###
        hsv_h=0.2,
        hsv_s=0.2,
        hsv_v=0.2,
        flipud=0.5,
        fliplr=0.5,
        ## Close strong augmentations ##
        mixup=0,
        cutmix=0,
        copy_paste=0,
        #########################
        project="runs",
        name="yolo8x_2021_pl"
    )
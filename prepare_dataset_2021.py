import argparse, os, json, shutil, random
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

def main(folds=5, seed=42):
    
    df1 = pd.read_csv("gwhd_2021/competition_train.csv")
    df2 = pd.read_csv("gwhd_2021/competition_val.csv")

    df = pd.concat([df1, df2], ignore_index=True)


    # 建立資料夾
    for split in ["train", "val"]:
        os.makedirs(f"datasets/2021/images/{split}", exist_ok=True)
        os.makedirs(f"datasets/2021/labels/{split}", exist_ok=True)
    X = df["image_name"]
    y = df["domain"]

    
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

        if fold >= 1:
            break

        filenames = df.iloc[train_idx]["image_name"]
        labels = df.iloc[train_idx]["BoxesString"]

        for filename, label in tqdm(zip(filenames, labels), total=len(filenames)):
            image_id = filename.split(".")[0]
            boxes = label.split(";")
            shutil.copyfile(f"gwhd_2021/images/{filename}", f"datasets/2021/images/train/{filename}")

            IMAGE_WIDTH = 1024
            IMAGE_HEIGHT = 1024

            def convert_box(xmin, ymin, xmax, ymax):
                
                x_center = (xmin + xmax) / 2 / IMAGE_WIDTH
                y_center = (ymin + ymax) / 2 / IMAGE_HEIGHT
                width = (xmax - xmin) / IMAGE_WIDTH
                height = (ymax - ymin) / IMAGE_HEIGHT
                return x_center, y_center, width, height
            

            with open(f"datasets/2021/labels/train/{image_id}.txt", "w") as f:
                if boxes[0] != "no_box":
                    for box in boxes:
                        xmin, ymin, xmax, ymax = map(int, box.strip().split())
                        x_c, y_c, w, h = convert_box(xmin, ymin, xmax, ymax)
                        f.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

        filenames = df.iloc[val_idx]["image_name"]
        labels = df.iloc[val_idx]["BoxesString"]

        for filename, label in tqdm(zip(filenames, labels), total=len(filenames)):
            image_id = filename.split(".")[0]
            boxes = label.split(";")
            shutil.copyfile(f"gwhd_2021/images/{filename}", f"datasets/2021/images/val/{filename}")

            with open(f"datasets/2021/labels/val/{image_id}.txt", "w") as f:
                if boxes[0] != "no_box":
                    for box in boxes:
                        xmin, ymin, xmax, ymax = map(int, box.strip().split())

                        x_c, y_c, w, h = convert_box(xmin, ymin, xmax, ymax)
                        f.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    yaml_dict = {
        "train": f"datasets/2021/images/train",
        "val":   f"datasets/2021/images/val",
        "nc": 1,
        "names": ["wheat_head"]
    }

    import yaml as pyaml
    
    with open(f"./wheat_2021.yaml", "w") as f:
        pyaml.dump(yaml_dict, f)
    print("✅ wheat_2021.yaml created.")

if __name__ == "__main__":
    main()

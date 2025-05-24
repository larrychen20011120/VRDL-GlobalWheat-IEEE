import numpy as np
from ultralytics import YOLO

# 1. 載入模型
model = YOLO("VDL_best.pt")
model.fuse()
model.to(0).eval()

# 2. 掃描不同 conf
conf_list = np.linspace(0.10, 0.30, 20)
best_conf, best_map = None, -1.0

for conf in conf_list:
    r = model.val(
        data="GlobalWheat2020.yaml",
        imgsz=1024,
        batch=36,
        workers=18,           # DataLoader 多執行緒
        conf=conf,
        save=False,
        verbose=False,       # 關掉多餘列印
        augment=True,
    )
    # 0~5 ⇒ IoU 0.50–0.75，共 6 個點
    map50_75 = float(np.mean(r.box.maps[:6]))
    print(f"conf={conf:.2f} → AP@[0.5:0.75]={map50_75:.4f}")

    if map50_75 > best_map:
        best_conf, best_map = conf, map50_75

print(f"\nBest conf：{best_conf:.2f}，AP@[0.5:0.75]={best_map:.4f}")

# YOLOv12-Seg 完整度识别使用指南

## 标签格式
- 分割标注行：`<cls> <completeness> <orientation> x1 y1 x2 y2 ...`，坐标依旧是归一化多边形点。
- 检测（无分割）行：`<cls> <completeness> <orientation> cx cy w h`。
- 建议：
  - 完整度：0=不完整，1=较完整，2=完整（设置 `comp_nc=3`）
  - 朝向：0=背面，1=侧面，2=正面（设置 `orient_nc=3`）

## 训练开关
- 新增超参：
  - `comp_nc`: 完整度类别数（0 关闭分支，默认为 0）。
  - `comp`: 完整度损失权重，默认 1.0。
  - `orient_nc`: 朝向类别数（0 关闭分支）。
  - `orient`: 朝向损失权重。
- 训练命令示例（完整度3类 + 朝向3类）：
  ```bash
  yolo train task=segment model=yolo11-seg.yaml data=your.yaml epochs=100 imgsz=640 \
    comp_nc=3 orient_nc=3 comp=1.0 orient=1.0
  ```

## 利用现有分割数据生成完整度标签
下面的伪代码展示了如何基于已有分割标签生成完整/不完整样本，并写回新标签：
```python
import random
import yaml
from pathlib import Path
import cv2
import numpy as np

def load_paths(data_yaml):
    data = yaml.safe_load(Path(data_yaml).read_text())
    img_dir, label_dir = Path(data["train"]), Path(data["train"]).parent / "labels"
    return sorted(img_dir.glob("*.jpg")), sorted(label_dir.glob("*.txt"))

def add_completeness(line, completeness):
    parts = line.strip().split()
    if len(parts) > 6 and len(parts) % 2 == 1:  # 已含完整度
        return line
    return " ".join([parts[0], str(completeness)] + parts[1:]) + "\n"

def random_incomplete(poly, keep_ratio=(0.5, 0.8)):
    poly = np.array(poly, dtype=float).reshape(-1, 2)
    cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
    scale = random.uniform(*keep_ratio)
    return (cx + (poly[:, 0] - cx) * scale, cy + (poly[:, 1] - cy) * scale)

def process_one(img_path, label_path, out_img_dir, out_lbl_dir):
    img = cv2.imread(str(img_path))
    lines = label_path.read_text().strip().splitlines()
    full_lines, cropped_lines = [], []
    for ln in lines:
        pts = ln.split()[1:]
        # 完整样本
        full_lines.append(add_completeness(ln, 2))  # 完整
        # 合成不完整样本：收缩多边形
        xy = random_incomplete(pts)
        flat = " ".join(f"{x:.6f}" for xy_pair in zip(*xy) for x in xy_pair)
        cropped_lines.append(f"{ln.split()[0]} 0 1 {flat}\n")  # 不完整，朝向侧面示例
    # 写入
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img_dir / img_path.name), img)              # 完整图
    (out_lbl_dir / label_path.name).write_text("".join(full_lines))
    cv2.imwrite(str(out_img_dir / f"incomplete_{img_path.name}"), img)
    (out_lbl_dir / f"incomplete_{label_path.name}").write_text("".join(cropped_lines))

def run(data_yaml, out_root):
    imgs, labels = load_paths(data_yaml)
    out_img_dir = Path(out_root) / "images"
    out_lbl_dir = Path(out_root) / "labels"
    for img_path, lbl_path in zip(imgs, labels):
        process_one(img_path, lbl_path, out_img_dir, out_lbl_dir)

# run("data/your.yaml", "data/your_completeness")
```
说明：
- 对每个实例保留原始标注（完整度=1），并生成一个收缩多边形的版本作为不完整样本（完整度=0）。
- 如果已有规则可判断“完整/缺失”，替换 `random_incomplete` 逻辑即可（如用关键部件覆盖率、缺口检测等）。

## 训练数据组织
- 使用上面生成的新数据目录更新 data.yaml 的 `train`/`val` 路径。
- 确保标签文本已插入 `completeness` 列，否则新头会自动跳过损失项。

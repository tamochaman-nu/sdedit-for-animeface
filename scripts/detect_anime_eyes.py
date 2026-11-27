"""
変換の出力 (アニメ顔、整列後の 256px) から目の位置を求める。hysts/anime-face-detector (yolov3 の顔検出と
28 点のランドマーク) を使う。この検出器は mmcv などの重い依存があるため、移植先のイメージには入れず、
anime-face-alignment のイメージ (ANIME_FA_IMAGE) で動かす (docker-compose.yml の detect-eyes)。

28 点のうち 11-16 と 17-22 が両目の輪郭 (anime-face-alignment の util/part_crop/part_geometry.py 19 行目)。
検出器は 256px では小さいので --upscale 倍に拡大して検出し、座標を元の大きさに戻す。顔が複数見つかったときは
顔のスコアが最大のものを使う。

入力: --input_dir/<名前>.png (例 /data/output/convert/samples)
出力: --out (json) {"args": {...}, "images": {<名前>: null | {"face_score", "bbox": [x0,y0,x1,y1],
       "eyes": [{"points": [[x, y] x 6], "score": 点の信頼度の平均}, {...}]}}}
"""

import argparse
import json
import os

import cv2
import numpy as np

EYE_IDX = (list(range(11, 17)), list(range(17, 23)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="yolov3", choices=["yolov3", "faster-rcnn"])
    p.add_argument("--upscale", type=int, default=2)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    from anime_face_detector import create_detector

    det = create_detector(a.model, device=a.device)
    names = sorted(f for f in os.listdir(a.input_dir) if f.lower().endswith(".png"))
    images = {}
    for fn in names:
        im = cv2.imread(os.path.join(a.input_dir, fn))
        h, w = im.shape[:2]
        big = cv2.resize(im, (w * a.upscale, h * a.upscale), interpolation=cv2.INTER_CUBIC)
        preds = det(big)
        stem = os.path.splitext(fn)[0]
        if not preds:
            images[stem] = None
            continue
        pr = max(preds, key=lambda q: q["bbox"][4])
        kp = np.asarray(pr["keypoints"], dtype=np.float64)
        kp[:, :2] /= a.upscale
        images[stem] = {
            "face_score": float(pr["bbox"][4]),
            "bbox": [float(v) / a.upscale for v in pr["bbox"][:4]],
            "eyes": [{"points": kp[idx, :2].tolist(), "score": float(kp[idx, 2].mean())} for idx in EYE_IDX],
        }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"args": vars(a), "images": images}, f, indent=1)
    ok = [v for v in images.values() if v]
    print(f"{len(ok)} / {len(images)} 枚で顔を検出 -> {a.out}")


if __name__ == "__main__":
    main()

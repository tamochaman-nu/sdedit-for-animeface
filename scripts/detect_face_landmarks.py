"""
実写の顔の 68 点ランドマーク (face_alignment ライブラリ: SFD で顔検出 -> 2DFAN4) を求めて JSON に保存する。
scripts/align_faces.py (FFHQ 方式の整列) の前段。

実行: docker compose run --rm landmarks [opts]  (/data/input -> /data/output/landmarks.json)
重み (2DFAN4、s3fd) は torch hub のキャッシュ (HOST_CACHE_DIR の torch/hub/checkpoints/) から読む。
イメージで TORCHDYNAMO_DISABLE=1 にしている (face_alignment 1.5 は torch.compile を使うが、compile すると
カーネルが変わって元の実行 (eager) と数値がずれるため。Dockerfile の注記)。
画像ごとに最も大きい顔を 1 つ選ぶ。出力 (JSON):
  {"args": ..., "images": {<ファイル名>: {"size": [w, h], "box": [x0, y0, x1, y1, 検出の信頼度],
                                          "landmarks": [[x, y], ...68 点], "scores": [...68 点],
                                          "num_faces": 検出した顔の数} または null (顔が無い)}}
座標は元画像のピクセル中心基準 (左上の画素の中心が (0, 0))。ただし face_alignment は座標を整数に切り捨てるため、
平均で約 -0.5 画素の偏りがある (quad の一辺は約 500px 以上なので、256px の出力では 0.3px 未満)。
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    import face_alignment

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=args.device, face_detector="sfd")
    names = sorted(n for n in os.listdir(args.input_dir) if n.lower().endswith(EXTS))
    res = {}
    for n in names:
        img = np.asarray(Image.open(os.path.join(args.input_dir, n)).convert("RGB"))
        lms, scores, boxes = fa.get_landmarks_from_image(img, return_bboxes=True, return_landmark_score=True)
        if not lms:
            res[n] = None
            print(n, "顔なし", flush=True)
            continue
        k = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]))
        res[n] = {"size": [img.shape[1], img.shape[0]], "box": [float(v) for v in boxes[k]],
                  "landmarks": np.asarray(lms[k], dtype=float)[:, :2].tolist(),
                  "scores": np.asarray(scores[k], dtype=float).reshape(-1).tolist(), "num_faces": len(lms)}
        print(n, f"顔 {len(lms)} 個、ランドマークの平均の信頼度 {np.mean(scores[k]):.3f}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "images": res}, f)


if __name__ == "__main__":
    main()

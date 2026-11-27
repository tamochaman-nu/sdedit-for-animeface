"""
変換の出力 (整列後の 256px) の目を、目のデカール (3dgs-for-styletransfer の eye_decal.py の RGBA。写真の目を ILVR の
アニメの目のモデルで変換したもの) で描き直す (投影元にする 1 枚の仕上げ用)。

目の位置: scripts/detect_anime_eyes.py の json の 6 点 (0-2 上まぶた、3-5 下まぶた。左から右)。目尻と目頭は
  (0 と 3 の中点)・(2 と 5 の中点)、向きはその 2 点を結ぶ向き、幅はその距離、高さは 6 点の向きに垂直な広がり。
デカールの楕円: 切り出しの一辺を 1 とした中心 (0.5, 0.5) と半径 --decal_axes (eye_decal.py の decals.json の
  crop.mask.axes。既定は sample04 の値)。この楕円を、出力の目の幅 × --scale_w、高さ × --scale_h に写す (アフィン)。
左右: --decal_l は画像の左の目 (x の小さい方)、--decal_r は右の目に貼る。

出力 (--out_dir): 実行のディレクトリと同じ並び (scripts/paste_back_faces.py の --run_dir にそのまま使える)。
  --names の画像だけを入れる
  samples/<名前>.png  描き直した出力、weights/<名前>.png  貼ったアルファ、config.json  入力の config.json の outputs を
  --names に絞り、"eye_decals" (設定と、目ごとの写した中心・幅・高さ) を足したもの

  docker compose run --rm --entrypoint python shell scripts/paste_eye_decals.py --run_dir /data/output/convert \\
      --eyes_json /data/output/convert/anime_eyes.json --names Cam3_frame_00008 \\
      --decal_l <左の目のデカール>.png --decal_r <右の目のデカール>.png --out_dir /data/output/eyedecal
"""

import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image


def eye_frame(points):
    p = np.asarray(points, dtype=np.float64)
    a, b = (p[0] + p[3]) / 2, (p[2] + p[5]) / 2
    u = (b - a) / np.linalg.norm(b - a)
    v = np.array([-u[1], u[0]])
    if v[1] < 0:  # 画像の下向き
        v = -v
    proj = (p - p.mean(0)) @ v
    return p.mean(0), u, v, float(np.linalg.norm(b - a)), float(proj.max() - proj.min())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--eyes_json", required=True)
    p.add_argument("--names", required=True, help="描き直す名前 (拡張子なし、カンマ区切り)")
    p.add_argument("--decal_l", required=True)
    p.add_argument("--decal_r", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--decal_axes", default="0.27,0.1142", help="デカールの楕円の半径 (切り出しの一辺 = 1)")
    p.add_argument("--scale_w", type=float, default=1.0)
    p.add_argument("--scale_h", type=float, default=1.0)
    p.add_argument("--min_score", type=float, default=0.3)
    a = p.parse_args()
    if os.path.abspath(a.out_dir) == os.path.abspath(a.run_dir):
        raise SystemExit("--out_dir は --run_dir と別にする")
    ax, ay = (float(v) for v in a.decal_axes.split(","))
    with open(os.path.join(a.run_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(a.eyes_json) as f:
        eyes = json.load(f)["images"]
    decals = {s: np.asarray(Image.open(path).convert("RGBA"), dtype=np.float64) / 255
              for s, path in (("l", a.decal_l), ("r", a.decal_r))}
    want = [n for n in a.names.split(",") if n]
    outs = [o for o in cfg["outputs"] if os.path.splitext(o["input"])[0] in want]
    if len(outs) != len(want):
        raise SystemExit(f"config.json に無い名前: {sorted(set(want) - {os.path.splitext(o['input'])[0] for o in outs})}")
    for sub in ("samples", "weights"):
        os.makedirs(os.path.join(a.out_dir, sub), exist_ok=True)
    info = {}
    for o in outs:
        stem = os.path.splitext(o["input"])[0]
        img = np.asarray(Image.open(os.path.join(a.run_dir, o["sample"])).convert("RGB"), dtype=np.float64) / 255
        H, W = img.shape[:2]
        es = [e for e in (eyes.get(stem) or {}).get("eyes", []) if e["score"] >= a.min_score]
        if len(es) != 2:
            raise SystemExit(f"{stem}: 信頼できる目が {len(es)} 個 (2 個要る)")
        es.sort(key=lambda e: np.mean(np.asarray(e["points"])[:, 0]))
        wsum = np.zeros((H, W))
        info[stem] = []
        for side, e in zip("lr", es):
            d = decals[side]
            s = d.shape[0]
            c, u, v, w, h = eye_frame(e["points"])
            # デカールの (x, y) -> 出力: c + u * (x - s/2) * kx + v * (y - s/2) * ky
            kx, ky = w * a.scale_w / (2 * ax * s), h * a.scale_h / (2 * ay * s)
            A = np.stack([u * kx, v * ky], 1)
            M = np.concatenate([A, (c - A @ np.array([s / 2, s / 2]))[:, None]], 1)
            warped = cv2.warpAffine(d, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            warped = np.clip(warped, 0, 1)
            al = warped[..., 3:4]
            img = img * (1 - al) + warped[..., :3] * al
            wsum = np.maximum(wsum, al[..., 0])
            info[stem].append({"side": side, "center": c.tolist(), "width": w, "height": h})
        Image.fromarray(np.round(img * 255).astype(np.uint8)).save(os.path.join(a.out_dir, o["sample"]))
        Image.fromarray(np.round(wsum * 255).astype(np.uint8)).save(os.path.join(a.out_dir, "weights", stem + ".png"))
    cfg["outputs"] = outs
    cfg["eye_decals"] = {"args": vars(a), "source_run": a.run_dir, "eyes": info}
    with open(os.path.join(a.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"{len(outs)} 枚 -> {a.out_dir}")


if __name__ == "__main__":
    main()

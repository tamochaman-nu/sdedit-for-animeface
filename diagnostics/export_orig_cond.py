"""
元リポジトリ (ilvr-for-anime-faceparts) のコードを編集せずに読み込み、採用構成の ④ 入力の加工 (部位別ストローク ->
色の写像) の出力を、量子化していない float のまま .pt に書き出す (⑤ の回帰確認の入力)。

設定は元の実行の config.json の args から取る (scripts/sdedit_sample.py 160-191 行目と同じ呼び出し)。
元のコードは mpi4py と blobfile を import するので、元のイメージ (ilvr-anime-faceparts:latest) で動かす
(docker compose run --rm export-orig)。/orig は読み取り専用でマウントする。

あわせて、書き出した float を元と同じ丸めで 8 bit にしたものが、元の実行の refs_cond/*.png と一致するかを確かめる
(元の実行の後にコードが更新されているため。config.json 18:11、face_input_transform.py と sdedit_sample.py 20:30)。

出力 (--out): {"names": [...], "cond": [N x 3 x H x W] float32, "args": 元の args, "orig_root": ...}
"""

import argparse
import json
import os
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orig_root", default="/orig")
    p.add_argument("--run_dir", default="outputs/sample04_stroke/parts2eye/SD", help="元の実行 (orig_root からの相対)")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    sys.path.insert(0, a.orig_root)
    import torch as th
    from PIL import Image

    from guided_diffusion.face_input_transform import transform_input
    from guided_diffusion.ilvr_eye import list_images, load_image, read_cfg, to_uint8_hwc
    from guided_diffusion.ilvr_ref_transform import ColorTransfer

    run_dir = os.path.join(a.orig_root, a.run_dir)
    with open(os.path.join(run_dir, "config.json")) as f:
        conf = json.load(f)
    args = conf["args"]
    # 採用構成だけを扱う (ほかの加工や ILVR が有効なら止める)
    assert args["use_sdedit"] and not args["use_ilvr"] and not args["use_mask"] and not args["use_color_fix"], args
    assert args["abstract_iters"] == 0 and args["quant_levels"] == 0 and args["xdog_strength"] == 0, args
    assert args["stroke"] == "parts" and not args["ref_list"] and args["num_refs"] == 0, args

    def orig(path):
        return path if os.path.isabs(path) else os.path.join(a.orig_root, path)

    cfg = read_cfg(orig(args["cfg"]))
    image_size = int(cfg["IMAGE_SIZE"])
    base = orig(args["base_samples"])
    names = list_images(base)
    assert names == [o["input"] for o in conf["outputs"]], "入力の一覧が元の実行の outputs と違う"
    stems = [os.path.splitext(n)[0] for n in names]
    x0_all = th.stack([load_image(os.path.join(base, n), image_size) for n in names])
    ct = ColorTransfer(orig(args["color_stats"]), args["color_transfer"], args["color_strength"])
    parse_dir = orig(args["stroke_parse_dir"])
    parse_all = th.stack([th.from_numpy(np.asarray(Image.open(os.path.join(parse_dir, s + ".png")), dtype=np.int64))
                          for s in stems])
    # config.json には stroke_eye_style / stroke_iris_rgb が無い (それらのオプションより前の実行)。現行の既定 (dark) を使う
    cond_all = transform_input(
        x0_all, abstract_iters=0, quant_levels=0, xdog_strength=0.0, color_transfer=ct, device="cpu",
        stroke="parts", parse=parse_all, stroke_shade_levels=args["stroke_shade_levels"],
        stroke_eye_scale_w=args["stroke_eye_scale_w"], stroke_eye_scale_h=args["stroke_eye_scale_h"],
        stroke_eye_style=args.get("stroke_eye_style", "dark"))

    # 元の実行の refs_cond/*.png (同じ丸め) と比べる
    n_bad_img, max_d, n_bad_px = 0, 0, 0
    for o, c in zip(conf["outputs"], cond_all):
        ref = np.asarray(Image.open(os.path.join(run_dir, o["cond_png"])).convert("RGB"), dtype=np.int16)
        d = np.abs(to_uint8_hwc(c).astype(np.int16) - ref)
        if d.max() > 0:
            n_bad_img += 1
        max_d = max(max_d, int(d.max()))
        n_bad_px += int((d.max(-1) > 0).sum())
    print(f"refs_cond との比較: 一致しない画像 {n_bad_img} / {len(names)} 枚、差の最大 {max_d}、"
          f"一致しない画素 {n_bad_px} / {len(names) * image_size * image_size}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    # メモリの並びは変えずに保存する (torch.save は stride も保存する)。元の ④ の出力は numpy の HWC を permute した
    # channels_last の並びで、そのまま ⑤ のモデルに入る。.contiguous() で NCHW に並べ替えると cuDNN の畳み込みの
    # カーネルが変わり、出力が 1 階調ほどずれる (フェーズ 2 の回帰確認で確かめた)
    assert cond_all.dtype == th.float32, cond_all.dtype
    print(f"④ の出力の stride {cond_all.stride()} (channels_last: {cond_all.is_contiguous(memory_format=th.channels_last)})")
    th.save({"names": names, "cond": cond_all, "args": args, "orig_root": a.orig_root,
             "refs_cond_check": {"mismatched_images": n_bad_img, "max_abs_diff": max_d, "mismatched_pixels": n_bad_px}},
            a.out)
    print(f"{len(names)} 枚の float の入力 {tuple(cond_all.shape)} -> {a.out}")


if __name__ == "__main__":
    main()

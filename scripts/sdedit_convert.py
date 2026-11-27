"""
切り出した実写の顔 (scripts/align_faces.py の出力) を、④ 入力の加工 -> ⑤ SDEdit でアニメ調に変換する。
既定値は採用構成 (ilvr-for-anime-faceparts の outputs/sample04_stroke/parts2eye/SD/config.json の args):
  間引き 100 ステップ、t0_ratio 0.5 (間引き後の番号 49 = 元の過程の t 494 から 50 ステップ)、K=1、ILVR なし、
  マスク合成なし、clip_denoised=True、全入力で同じシード 0、バッチ 16。
ループは animeface/sdedit.py。scripts/sdedit_sample.py (元リポジトリ) の該当部分を書き写した。

④ 入力の加工 (animeface/face_input_transform.py。元の sdedit_sample.py 152-198 行目と同じ順序):
  --base_samples/<名前>.png を読み (center_crop_arr、[-1, 1])、--stroke_parse_dir/<名前>.png のラベルで部位別に塗る
  (--stroke_shade_levels 段の影、目の領域を --stroke_eye_scale_w x --stroke_eye_scale_h 倍)。
  前景の色を --color_transfer (統計 --color_stats、強さ --color_strength) でアニメ顔のデータに写す。
  加工した入力は量子化せずに float のまま ⑤ の開始点に使う (refs_cond/ の PNG は確認用)。
⑤ SDEdit (animeface/sdedit.py): x = q_sample(c0, t0-1) から t0 ステップ。t0 = max(1, int(--t0_ratio * T))。

入力 (回帰確認用):
  --cond_pt   ④ の出力を書き出した .pt ({"names": [...], "cond": [N x 3 x H x W]})。指定すると ④ を飛ばして、
              これを ⑤ の開始点に使う (diagnostics/export_orig_cond.py、--save_cond_pt)
  --save_cond_pt  ④ の出力を float のまま .pt に書き出す (メモリの並びも保つ)

出力 (--out_dir):
  samples/<入力名>.png   変換結果 ([-1, 1] を丸めた 8 bit)
  refs/<入力名>.png      前処理した入力 (④ をしたときだけ)
  refs_cond/<入力名>.png ④ の出力を 8 bit にしたもの (確認用。④ をしたときだけ)
  config.json            全てのオプション、導出した値 (開始の番号・元の過程の t・√ᾱ・SNR)、出力の一覧、時間
                         (scripts/paste_back_faces.py が outputs[].input と outputs[].sample を読む)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch as th
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animeface.color_transfer import COLOR_TRANSFER_CHOICES, ColorTransfer  # noqa: E402
from animeface.face_input_transform import transform_input  # noqa: E402
from animeface.image_io import list_images, load_image, save_png, str2bool  # noqa: E402
from animeface.model import create_diffusion_from_cfg, load_model, read_cfg  # noqa: E402
from animeface.sdedit import SampleRNG, sdedit_sample, sdedit_t0, start_info  # noqa: E402


def main():
    args = create_argparser().parse_args()
    device = th.device(args.device)

    cfg = read_cfg(args.cfg)
    model = load_model(cfg, args.model_path, device)
    diffusion = create_diffusion_from_cfg(cfg, args.timestep_respacing)
    image_size = int(cfg["IMAGE_SIZE"])
    T = diffusion.num_timesteps
    t0 = sdedit_t0(args.t0_ratio, T)
    start = t0 - 1

    os.makedirs(os.path.join(args.out_dir, "samples"), exist_ok=True)
    if args.cond_pt:
        d = th.load(args.cond_pt, map_location="cpu", weights_only=True)
        # メモリの並び (stride) は変えない。元の ④ の出力は channels_last の並びで、並びが変わると cuDNN のカーネルが
        # 変わって出力がずれる (diagnostics/export_orig_cond.py の注記)
        names, cond_all = list(d["names"]), d["cond"]
        stems = [os.path.splitext(n)[0] for n in names]
    else:
        assert args.base_samples and args.stroke_parse_dir, "④ には --base_samples と --stroke_parse_dir が要る"
        names = list_images(args.base_samples)
        stems = [os.path.splitext(n)[0] for n in names]
        x0_all = th.stack([load_image(os.path.join(args.base_samples, n), image_size) for n in names])
        ct = (ColorTransfer(args.color_stats, args.color_transfer, args.color_strength)
              if args.color_transfer != "none" else None)
        parse_all = th.stack([th.from_numpy(np.asarray(Image.open(os.path.join(args.stroke_parse_dir, s + ".png")),
                                                       dtype=np.int64)) for s in stems])
        assert tuple(parse_all.shape[1:]) == tuple(x0_all.shape[2:]), (parse_all.shape, x0_all.shape)
        cond_all = transform_input(x0_all, parse_all, color_transfer=ct, stroke_shade_levels=args.stroke_shade_levels,
                                   stroke_eye_scale_w=args.stroke_eye_scale_w,
                                   stroke_eye_scale_h=args.stroke_eye_scale_h)
        for sub, imgs in (("refs_cond", cond_all), ("refs", x0_all)):
            os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)
            for stem, c in zip(stems, imgs):
                save_png(c, os.path.join(args.out_dir, sub, f"{stem}.png"))
        if args.save_cond_pt:
            os.makedirs(os.path.dirname(args.save_cond_pt) or ".", exist_ok=True)
            th.save({"names": names, "cond": cond_all, "args": vars(args)}, args.save_cond_pt)
    assert cond_all.dtype == th.float32, cond_all.dtype
    assert tuple(cond_all.shape[1:]) == (3, image_size, image_size), cond_all.shape

    outputs = []
    if device.type == "cuda":
        th.cuda.reset_peak_memory_stats(device)
        th.cuda.synchronize(device)
    t_start = time.time()
    for st in range(0, len(names), args.batch_size):
        idx = list(range(st, min(st + args.batch_size, len(names))))
        c0 = cond_all[idx].to(device)
        rng = SampleRNG([args.seed] * len(idx), device)  # 全入力で同じノイズ列
        x = sdedit_sample(diffusion, model, c0, t0, rng=rng, clip_denoised=args.clip_denoised,
                          progress=args.progress)
        for j, i in enumerate(idx):
            out = os.path.join("samples", f"{stems[i]}.png")
            save_png(x[j], os.path.join(args.out_dir, out))
            o = {"input": names[i], "sample": out, "noise_seed": args.seed}
            if not args.cond_pt:
                o.update(ref_png=os.path.join("refs", f"{stems[i]}.png"),
                         cond_png=os.path.join("refs_cond", f"{stems[i]}.png"))
            outputs.append(o)
    if device.type == "cuda":
        th.cuda.synchronize(device)
    elapsed = time.time() - t_start

    config = {
        "args": vars(args),
        "derived": {
            "num_timesteps": T,
            "sdedit_t0_steps": t0,
            "sdedit_start": start_info(diffusion, start),
            "input": "cond_pt" if args.cond_pt else "transform_input",
            "cond_channels_last": cond_all.is_contiguous(memory_format=th.channels_last),
        },
        "train_cfg": cfg,
        "outputs": outputs,
        "elapsed_sec": round(elapsed, 3),
        "peak_vram_mb": round(th.cuda.max_memory_allocated(device) / 2 ** 20, 1) if device.type == "cuda" else None,
        "torch": th.__version__,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    s = config["derived"]["sdedit_start"]
    print(f"{len(outputs)} 枚 -> {args.out_dir} ({elapsed:.1f}s)。開始: 間引き後の番号 {s['start_index']}、"
          f"元の過程の t {s['t_original']}、√ᾱ {s['sqrt_alpha_bar']:.4f}、SNR {s['snr']:.4f}")


def create_argparser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    def flag(name, default, help_):
        p.add_argument(f"--{name}", type=str2bool, nargs="?", const=True, default=default, help=help_)

    p.add_argument("--model_path", required=True)
    p.add_argument("--cfg", required=True, help="学習時の train.cfg (ddpm/cfg/*.infer.cfg)")
    p.add_argument("--out_dir", required=True)
    # ④ 入力の加工
    p.add_argument("--base_samples", default="", help="切り出した顔のディレクトリ (scripts/align_faces.py の出力)")
    p.add_argument("--stroke_parse_dir", default="", help="顔パーシングのラベル (scripts/parse_faces.py の出力)")
    p.add_argument("--stroke_shade_levels", type=int, default=2, help="肌・首・髪・服の影の段数 (1 で影なし)")
    p.add_argument("--stroke_eye_scale_w", type=float, default=1.2, help="目の領域の横の倍率")
    p.add_argument("--stroke_eye_scale_h", type=float, default=1.8, help="目の領域の縦の倍率")
    p.add_argument("--color_transfer", default="lab_meanstd_l", choices=COLOR_TRANSFER_CHOICES)
    p.add_argument("--color_stats", default="lists/face_color_stats.json", help="色の写像の統計")
    p.add_argument("--color_strength", type=float, default=1.0)
    p.add_argument("--cond_pt", default="", help="④ の float の出力 (.pt)。指定すると ④ を飛ばす")
    p.add_argument("--save_cond_pt", default="", help="④ の float の出力を書き出す .pt")
    p.add_argument("--timestep_respacing", default="100")
    p.add_argument("--t0_ratio", type=float, default=0.5, help="t0 = max(1, int(t0_ratio * T))。番号 t0-1 から t0 ステップ")
    p.add_argument("--seed", type=int, default=0, help="乱数のシード (全ての入力で同じノイズ列)")
    flag("clip_denoised", True, "x0 の予測を [-1, 1] に切り詰める")
    flag("progress", False, "")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    main()

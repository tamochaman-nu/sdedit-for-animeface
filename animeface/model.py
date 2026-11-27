"""
学習時の train.cfg (KEY="VALUE" 形式) から MPUNetModel と拡散過程を作り、チェックポイントを読む。

ilvr-for-anime-faceparts (87152b3) からの書き写し:
  read_cfg, _bool, create_model_from_cfg, create_diffusion_from_cfg, load_model  <- guided_diffusion/ilvr_eye.py 38-103
  create_gaussian_diffusion, NUM_CLASSES                                          <- guided_diffusion/script_util.py 8, 386-424
script_util.py は unet.py (公式の UNetModel) を import するので、要る関数だけをここに写した (計算は同じ)。
"""

import re

import torch as th

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet_mp import MPUNetModel

NUM_CLASSES = 1000


def read_cfg(path):
    """学習フォークの scripts/load_cfg.sh (save_cfg) が書く KEY="VALUE" 形式の train.cfg を読む。"""
    cfg = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"^\s*([A-Z_][A-Z0-9_]*)=(.*)$", line.split("#")[0].rstrip())
            if m:
                cfg[m.group(1)] = m.group(2).strip().strip('"')
    return cfg


def _bool(s):
    return str(s).lower() in ("true", "1", "yes")


def create_model_from_cfg(cfg):
    """
    学習フォーク (tamochaman-nu/guided-diffusion@5bb6a5c) の script_util.create_model の arch="mp" 分岐と
    同じ引数で MPUNetModel を作る。推論なので fp32 (use_bf16 / use_fp16 は使わない)。
    LEARN_SIGMA=True なら出力は 6 チャネル (前半 3 が epsilon、後半 3 が分散の補間係数。
    分け方は gaussian_diffusion.GaussianDiffusion.p_mean_variance の LEARNED_RANGE の分岐)。
    """
    assert cfg["ARCH"] == "mp", f"ARCH={cfg['ARCH']}: MPUNetModel 用の cfg ではない"
    assert not _bool(cfg.get("ATTR_COND", "False")), "属性条件付きのモデルには対応していない"
    image_size = int(cfg["IMAGE_SIZE"])
    channel_mult = cfg.get("CHANNEL_MULT", "")
    assert channel_mult, "CHANNEL_MULT が空 (image_size ごとの既定値は使わず、cfg に明示されている前提)"
    clip_act = float(cfg.get("MP_CLIP_ACT", "256"))
    return MPUNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=int(cfg["NUM_CHANNELS"]),
        out_channels=6 if _bool(cfg["LEARN_SIGMA"]) else 3,
        num_res_blocks=int(cfg["NUM_RES_BLOCKS"]),
        attention_resolutions=tuple(image_size // int(r) for r in cfg["ATTENTION_RESOLUTIONS"].split(",")),
        dropout=float(cfg["DROPOUT"]),
        channel_mult=tuple(int(m) for m in channel_mult.split(",")),
        num_classes=NUM_CLASSES if _bool(cfg["CLASS_COND"]) else None,
        use_checkpoint=False,
        use_fp16=False,
        use_bf16=False,
        num_heads=int(cfg["NUM_HEADS"]),
        num_head_channels=int(cfg["NUM_HEAD_CHANNELS"]),
        diffusion_steps=int(cfg["DIFFUSION_STEPS"]),
        noise_schedule=cfg["NOISE_SCHEDULE"],
        clip_act=clip_act if clip_act > 0 else None,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def create_diffusion_from_cfg(cfg, timestep_respacing=""):
    # 学習フォークの run_train.sh は diffusion_steps / noise_schedule 以外を渡さないので、残りは script_util の既定値
    return create_gaussian_diffusion(
        steps=int(cfg["DIFFUSION_STEPS"]),
        learn_sigma=_bool(cfg["LEARN_SIGMA"]),
        sigma_small=False,
        noise_schedule=cfg["NOISE_SCHEDULE"],
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing=timestep_respacing,
    )


def load_model(cfg, model_path, device):
    model = create_model_from_cfg(cfg)
    model.load_state_dict(th.load(model_path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()

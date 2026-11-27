"""
SDEdit (Meng et al., ICLR 2022) の逆過程のループ。間引いた拡散過程 (SpacedDiffusion) と、学習した分散 (LEARNED_RANGE) を使う。

ilvr-for-anime-faceparts (87152b3、作業ツリー) からの書き写し:
  sdedit_t0, start_info  <- guided_diffusion/sdedit.py 27-42
  sdedit_sample          <- guided_diffusion/sdedit.py 50-113 の sdedit_ilvr_sample のうち、resizers=None (ILVR なし)、
                            mask=None (マスク合成なし)、x_init=None、intermediate=None の経路だけ。計算と乱数の消費順は同じ
  SampleRNG              <- guided_diffusion/ilvr_eye.py 159-168

ステップ番号 i はすべて間引き後の番号 0..T-1 (T = diffusion.num_timesteps)。diffusion.q_sample(x, i) は間引き後の
alphas_cumprod[i] (= 元の過程の alphas_cumprod[timestep_map[i]]) を使い、モデルには p_mean_variance の中で
元の番号 timestep_map[i] が渡される (respace._WrappedModel)。MPUNetModel はこの元の番号から log(sigma) を引くので、
間引き後の番号をモデルに直接渡してはいけない。番号 i の 1 ステップは x_i から x_{i-1} を作る。

公式 SDEdit (runners/image_editing.py) との違いは README.md の表を参照。
"""

import torch as th


def sdedit_t0(t0_ratio, num_timesteps):
    """t0 = max(1, int(t0_ratio * T)) を 1..T に収めて返す (回すステップ数。開始の番号は t0 - 1)。"""
    return min(max(1, int(t0_ratio * num_timesteps)), num_timesteps)


def start_info(diffusion, start):
    """開始の番号 start (間引き後) の、元の過程の t、√ᾱ、SNR = ᾱ / (1 - ᾱ)。"""
    a = float(diffusion.alphas_cumprod[start])
    t_orig = int(diffusion.timestep_map[start]) if hasattr(diffusion, "timestep_map") else start
    return {"start_index": start, "t_original": t_orig, "sqrt_alpha_bar": a ** 0.5, "snr": a / (1 - a)}


class SampleRNG:
    """サンプルごとの Generator から標準正規乱数を引く。バッチの組み方を変えても各サンプルの乱数は同じ。"""

    def __init__(self, seeds, device):
        self.device = th.device(device)
        self.generators = [th.Generator(device=self.device).manual_seed(int(s)) for s in seeds]

    def __call__(self, shape):
        assert shape[0] == len(self.generators), (shape, len(self.generators))
        return th.stack([th.randn(shape[1:], generator=g, device=self.device) for g in self.generators])


def sdedit_sample(diffusion, model, x0, t0, rng=None, clip_denoised=True, progress=False):
    """
    x0 ([N x 3 x H x W], [-1, 1]): 入力 (量子化していない float)。
    t0 (1..T): 回すステップ数。x = q_sample(x0, t0-1) から始めて、i = t0-1, ..., 0 を回す。
    rng: shape -> 標準正規乱数。省略するとグローバルな乱数。乱数は開始時に 1 回、各ステップに 1 回引く
         (i = 0 でも引いて 0 を掛ける)。
    """
    shape = tuple(x0.shape)
    device = x0.device
    num_timesteps = diffusion.num_timesteps
    if rng is None:
        def rng(shape):
            return th.randn(*shape, device=device)

    assert 1 <= t0 <= num_timesteps, (t0, num_timesteps)
    start = t0 - 1
    img = diffusion.q_sample(x0, th.full((shape[0],), start, device=device, dtype=th.long), rng(shape))

    indices = list(range(start + 1))[::-1]
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices, leave=False)
    for i in indices:
        t = th.full((shape[0],), i, device=device, dtype=th.long)
        with th.no_grad():
            # diffusion.p_sample と同じ計算 (乱数だけ rng から引く。t == 0 でも乱数は引き、0 を掛ける)
            out = diffusion.p_mean_variance(model, img, t, clip_denoised=clip_denoised, model_kwargs={})
            nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
            x = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * rng(shape)
        img = x
    return img

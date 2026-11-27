"""
画像ごとの「アニメらしさ」の指標 (推論だけ)。Inception-v3 (torch-fidelity の FID/KID 用の重み) の 2048 次元の
プール特徴に、アニメ顔のデータと実写の顔 (FFHQ) を見分けるロジスティック回帰をかけ、ロジットを返す。

元リポジトリ (ilvr-for-anime-faceparts) からの書き写し (学習の fit は写していない):
  AnimeScore, features    <- guided_diffusion/anime_score.py 23-28, 83-98
  InceptionFeatures       <- guided_diffusion/ilvr_metrics.py 76-91
分類器の重みは diagnostics/assets/anime_score.pt (元の diagnostics/outputs/anime_score/anime_score.pt の複製。
学習の設定と正解率は中の "info")。Inception の重みは torch hub のキャッシュ (~/.cache/torch/hub/checkpoints、
docker-compose.yml で .cache をマウント) から読み、無ければ torch-fidelity が取得する。
ただし分類器が見分けやすい手がかり (質感・線・色) に引っ張られる点に注意。
"""

import torch as th


class InceptionFeatures:
    """torch-fidelity の Inception-v3 (FID/KID の標準の重み) の 2048 次元のプール特徴。入力は内部で 299px に拡大される。"""

    def __init__(self, device):
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

        self.device = device
        self.model = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).to(device).eval()

    def __call__(self, x, batch_size=500):
        feats = []
        with th.no_grad():
            for i in range(0, len(x), batch_size):
                u8 = ((x[i:i + batch_size].clamp(-1, 1) + 1) * 127.5).round().to(th.uint8).to(self.device)
                feats.append(self.model(u8)[0].double().cpu())
        return th.cat(feats)


def features(extractor, images, batch_size=64):
    """images: [N x 3 x H x W] in [-1, 1] (CPU 可) -> [N x 2048] (CPU)"""
    out = []
    for i in range(0, len(images), batch_size):
        out.append(extractor(images[i:i + batch_size]).float().cpu())
    return th.cat(out)


class AnimeScore:
    def __init__(self, path, device):
        d = th.load(path, map_location="cpu", weights_only=False)
        self.m = {k: v.to(device) for k, v in d["model"].items()}
        self.info = d["info"]
        self.ext = InceptionFeatures(device)
        self.device = device

    def logit(self, images):
        """images: [N x 3 x H x W] in [-1, 1] -> ロジット [N] (CPU)。確率は 0 や 1 に張り付きやすいので比較はこちらで"""
        f = features(self.ext, images).to(self.device)
        return (((f - self.m["mu"]) / self.m["sd"]) @ self.m["w"] + self.m["b"]).cpu()

    def __call__(self, images):
        """アニメである確率 [N] (CPU)"""
        return th.sigmoid(self.logit(images))

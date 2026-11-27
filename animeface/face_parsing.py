"""
face_parsing.py
実写人物画像向けのセマンティックセグメンテーション（顔パージング）。

BiSeNet（ResNet18バックボーン、CelebAMask-HQ 19クラスで学習済み）により、画像中の
背景・肌・髪・帽子・服飾などの領域をピクセル単位で分類する。util.ffhq_face_alignmentの
remove_background()がこのモジュールのregion_masks()を呼び、背景領域の除去に使う。

アーキテクチャはhttps://github.com/yakhyo/face-parsing （MIT License,
Copyright (c) 2024 Yakhyokhuja Valikhujaev）のmodels/bisenet.py・models/resnet.pyを、
学習済みチェックポイントの重みキー（fpn.backbone.*, fpn.arm16.*, ffm.*, conv_out.*等）を
変えずに移植したもの。ただしResNetの分類ヘッド（avgpool/fc）は推論に使わないため省略し、
チェックポイント側にのみ存在するfpn.backbone.fc.*キーはロード時に無視する
（strict=Falseとした上で、無視されたキーがfc以外に無いことをアサートで確認する）。
学習済み重み自体はアーキテクチャと別管理で、初回使用時に自動ダウンロードする。
"""

import os

import torch
import torch.nn.functional as F
from torch import nn

_WEIGHTS_URL = "https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.pt"
_WEIGHTS_PATH = os.path.join(os.path.expanduser("~"), ".cache", "face_parsing", "resnet18.pt")
_NUM_CLASSES = 19
_INPUT_SIZE = 512

# BiSeNetはImageNet事前学習ベースのバックボーンを使うため、推論時もImageNet統計で正規化する
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# CelebAMask-HQ 19クラスの内訳（学習データ側の定義。yakhyo/face-parsingのutils/common.py
# のATTRIBUTES、および本体チェックポイントの出力チャネル順と一致することを確認済み）
#   0: background   1: skin        2: l_brow    3: r_brow    4: l_eye
#   5: r_eye        6: eye_g       7: l_ear     8: r_ear     9: ear_r
#   10: nose        11: mouth      12: u_lip    13: l_lip    14: neck
#   15: neck_l      16: cloth      17: hair     18: hat
#
# 19クラスをそのまま使わず、用途に応じて必要な粗い領域グループに束ねたもの
REGION_GROUPS = {
    "background": [0],
    "skin": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "hair": [17],
    "hat": [18],
    "clothing": [14, 15, 16],
}


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class _BasicBlock(nn.Module):
    """ResNet18の基本ブロック（3x3conv x2 + 残差接続）"""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _ResNet18Backbone(nn.Module):
    """標準的なResNet18。conv1→maxpool→layer1..4でstride8/16/32の3段階特徴マップを出力する。
    分類ヘッド（avgpool/fc）は推論で使わないため未実装（チェックポイントのfpn.backbone.fc.*
    キーは呼び出し側でstrict=Falseにして無視する）。
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

    @staticmethod
    def _make_layer(in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [_BasicBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(_BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        feat8 = self.layer2(x)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat8, feat16, feat32


class _AttentionRefinementModule(nn.Module):
    """グローバル平均プーリング+sigmoidゲートでチャネル注意を適用する"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = _ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=1)
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.conv_block(x)
        pool = F.avg_pool2d(feat, feat.shape[2:])
        return feat * self.attention(pool)


class _ContextPath(nn.Module):
    """backboneの出力に注意機構付きの特徴融合・アップサンプリングを適用し、コンテキスト特徴を集約する"""

    def __init__(self):
        super().__init__()
        self.backbone = _ResNet18Backbone()
        self.arm16 = _AttentionRefinementModule(256, 128)
        self.arm32 = _AttentionRefinementModule(512, 128)
        self.conv_head32 = _ConvBNReLU(128, 128, kernel_size=3, stride=1)
        self.conv_head16 = _ConvBNReLU(128, 128, kernel_size=3, stride=1)
        self.conv_avg = _ConvBNReLU(512, 128, kernel_size=1, stride=1)

    def forward(self, x):
        feat8, feat16, feat32 = self.backbone(x)
        h8, w8 = feat8.shape[2:]
        h16, w16 = feat16.shape[2:]
        h32, w32 = feat32.shape[2:]

        avg = F.avg_pool2d(feat32, feat32.shape[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (h32, w32), mode="nearest")

        feat32_sum = self.arm32(feat32) + avg_up
        feat32_up = self.conv_head32(F.interpolate(feat32_sum, (h16, w16), mode="nearest"))

        feat16_sum = self.arm16(feat16) + feat32_up
        feat16_up = self.conv_head16(F.interpolate(feat16_sum, (h8, w8), mode="nearest"))

        return feat8, feat16_up, feat32_up


class _FeatureFusionModule(nn.Module):
    """空間特徴とコンテキスト特徴を結合し、チャネル注意付きの残差融合を行う"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = _ConvBNReLU(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(out_channels, out_channels // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        feat = self.conv_block(torch.cat([fsp, fcp], dim=1))
        attention = F.avg_pool2d(feat, feat.shape[2:])
        attention = self.sigmoid(self.conv2(self.relu(self.conv1(attention))))
        return feat * attention + feat


class _BiSeNetOutput(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.conv_block = _ConvBNReLU(in_channels, mid_channels, kernel_size=3, stride=1)
        self.conv = nn.Conv2d(mid_channels, num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv(self.conv_block(x))


class _BiSeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fpn = _ContextPath()
        self.ffm = _FeatureFusionModule(in_channels=256, out_channels=256)
        self.conv_out = _BiSeNetOutput(in_channels=256, mid_channels=256, num_classes=num_classes)
        # 学習時専用の補助出力ヘッド。推論経路では使わないが、チェックポイントの重みキー
        # （conv_out16.*, conv_out32.*）に対応させるため定義だけ残す
        self.conv_out16 = _BiSeNetOutput(in_channels=128, mid_channels=64, num_classes=num_classes)
        self.conv_out32 = _BiSeNetOutput(in_channels=128, mid_channels=64, num_classes=num_classes)

    def forward(self, x):
        h, w = x.shape[2:]
        feat_res8, feat_cp8, _feat_cp16 = self.fpn(x)
        feat_fuse = self.ffm(feat_res8, feat_cp8)
        logits = self.conv_out(feat_fuse)
        return F.interpolate(logits, (h, w), mode="bilinear", align_corners=True)


_model = None
_model_device = None


def _download_weights():
    os.makedirs(os.path.dirname(_WEIGHTS_PATH), exist_ok=True)
    torch.hub.download_url_to_file(_WEIGHTS_URL, _WEIGHTS_PATH)


def _get_model(device):
    """モデルをグローバルにキャッシュする（デバイスが変わった場合のみ再構築する）"""
    global _model, _model_device
    if _model is not None and _model_device == device:
        return _model

    if not os.path.exists(_WEIGHTS_PATH):
        _download_weights()

    model = _BiSeNet(_NUM_CLASSES)
    state_dict = torch.load(_WEIGHTS_PATH, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not missing, f"BiSeNet重みのロードに失敗しました（missing keys）: {missing}"
    assert all(k.startswith("fpn.backbone.fc.") for k in unexpected), (
        f"BiSeNet重みのロード時に想定外のunexpected keysがありました: {unexpected}"
    )

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    _model = model
    _model_device = device
    return model


@torch.no_grad()
def parse_class_map(img01, device):
    """img01: (3, H, W), [0,1]。任意解像度 → 512x512にリサイズして推論 → NEARESTで
    元解像度に戻す（クラス境界を滑らかにせず保持するため）。
    戻り値: (H, W) のクラスIDテンソル（long, 0-18）。
    """
    model = _get_model(device)
    h, w = img01.shape[1:]

    x = img01.unsqueeze(0).to(device)
    x = F.interpolate(x, size=(_INPUT_SIZE, _INPUT_SIZE), mode="bilinear", align_corners=False)
    x = (x - _IMAGENET_MEAN.to(device)) / _IMAGENET_STD.to(device)

    logits = model(x)
    class_map = logits.argmax(dim=1, keepdim=True).float()
    class_map = F.interpolate(class_map, size=(h, w), mode="nearest")
    return class_map.squeeze(0).squeeze(0).long()


@torch.no_grad()
def parse_class_map_batch(img01_batch, device):
    """parse_class_mapのバッチ版。img01_batch: (B, 3, H, W), [0,1]。戻り値: (B, H, W) のクラスID（long）"""
    model = _get_model(device)
    h, w = img01_batch.shape[2:]
    x = F.interpolate(img01_batch.to(device), size=(_INPUT_SIZE, _INPUT_SIZE), mode="bilinear", align_corners=False)
    x = (x - _IMAGENET_MEAN.to(device)) / _IMAGENET_STD.to(device)
    class_map = model(x).argmax(dim=1, keepdim=True).float()
    return F.interpolate(class_map, size=(h, w), mode="nearest").squeeze(1).long()


def region_masks(img01, device):
    """img01: (3, H, W), [0,1]。
    戻り値: {region_name: (H, W) bool} の辞書。該当ピクセルが0件の領域（帽子・メガネ等）も
    全Falseマスクとして必ずキーに含まれる。
    """
    class_map = parse_class_map(img01, device)
    masks = {}
    for name, class_ids in REGION_GROUPS.items():
        mask = torch.zeros_like(class_map, dtype=torch.bool)
        for class_id in class_ids:
            mask |= class_map == class_id
        masks[name] = mask.cpu().numpy()
    return masks

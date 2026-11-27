# 実写の顔 -> アニメ顔の変換 (animeface/、scripts/) の実行環境。使い方は docker-compose.yml の冒頭と README.md。
#
# ベースは元の実行 (ilvr-for-anime-faceparts の ilvr-anime-faceparts:latest) と同じ PyTorch 2.4.1 + CUDA 12.4。
# ⑤ の回帰確認で画素単位の一致を目指すため。ホストのドライバに合わせて切り替える:
#   >= 550: cuda12.4 / >= 530: cuda12.1 / >= 520: cuda11.8
ARG BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

# ベースの Python パッケージの版 (torch など) を制約にして、追加の依存で更新されないようにする
COPY requirements-animeface.txt /tmp/requirements-animeface.txt
RUN pip freeze | grep -E '^[A-Za-z0-9_.-]+==' | grep -v -i -E '^(scipy|scikit-image)==' > /tmp/base-constraints.txt \
    && pip install -c /tmp/base-constraints.txt -r /tmp/requirements-animeface.txt \
    # face-alignment は opencv-python に依存するが、上の headless 版と同じ cv2 の名前空間を上書きするので --no-deps で入れる
    # (anime-face-alignment の Dockerfile 26 行目と同じ扱い)。残りの依存 (torch, numpy, scipy, scikit-image, numba,
    # tqdm, packaging) は上とベースイメージで入っている
    && pip install --no-deps face-alignment==1.5.0 \
    && python -c "import cv2, numba, skimage, scipy, face_alignment; print('cv2', cv2.__version__, 'skimage', skimage.__version__)"

# コンテナはホストのユーザー (docker-compose.yml の user。イメージに無い UID でもよい) で動かすので、
# どの UID でも書ける HOME を用意する。重みのキャッシュ (torch hub と BiSeNet) は ~/.cache に置き、
# docker-compose.yml でリポジトリ内の .cache をここにマウントする
RUN mkdir -p /home/app/.cache && chmod -R 1777 /home/app
ENV HOME=/home/app USER=app

# face_alignment 1.5 は torch.compile を使う。compile するとカーネルが変わって元の実行 (eager) と数値がずれるので無効にする
# (元の実行も TORCHDYNAMO_DISABLE=1。compile のキャッシュの読み書きに失敗しても face_alignment は eager で続ける)
ENV TORCHDYNAMO_DISABLE=1

# リポジトリは /app にバインドマウントする (コードを変えても再ビルドは不要)
WORKDIR /app
ENV PYTHONPATH=/app

CMD ["/bin/bash"]

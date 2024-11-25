FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# 作業ディレクトリを設定
WORKDIR /workspace

# 必要なツールをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# CUDAの動作確認用に必要なコマンドをインストール（必要に応じて）
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-tools-12-4 \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt update && apt install -y ffmpeg

# 必要な環境変数を設定
ENV PATH /usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

# コンテナ起動時のデフォルトコマンド
CMD ["bash"]

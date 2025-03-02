"""
従来のVAE用 predict_model.py を置き換え。
今度は学習済みの cGAN Generator を使い、条件を与えて間取りを生成する例
"""
import os
import torch
import numpy as np
from glob import glob

from .gan import Generator

# 定数を直接定義
NOISE_DIM = 64
COND_DIM = 1
OUT_CHANNELS = 19

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_available_models(models_dir="models"):
    """利用可能なモデルファイルのリストを取得"""
    if not os.path.exists(models_dir):
        return []
    # generator_*.pthファイルを検索
    model_files = glob(os.path.join(models_dir, "generator_*.pth"))
    # ファイル名のみを返す
    return [os.path.basename(f) for f in model_files]

def load_generator(gen_path="models/generator_ep100.pth", noise_dim=NOISE_DIM, cond_dim=COND_DIM, out_channels=OUT_CHANNELS):
    netG = Generator(noise_dim=noise_dim, cond_dim=cond_dim, out_channels=out_channels)
    netG.load_state_dict(torch.load(gen_path, map_location=device))
    netG.to(device)
    netG.eval()
    return netG

def generate_floorplans(gen_model, num_samples=5, cond_value=0.2):
    """
    gen_model: 学習済みGenerator
    num_samples: 生成枚数
    cond_value : 簡易的に 1次元の条件を指定(例: 0.2)
    戻り値: [ (H,W) array(ラベルID or ロジットargmax), ... ]
    """
    results = []
    with torch.no_grad():
        for i in range(num_samples):
            # ノイズ
            z = torch.randn(1, NOISE_DIM, device=device)
            cond = torch.tensor([[cond_value]], dtype=torch.float, device=device)  # (1,1)
            fake_logits = gen_model(z, cond)
            # shape=(1,out_channels,64,64)
            # argmax
            pred = fake_logits.argmax(dim=1).squeeze(0).cpu().numpy()
            results.append(pred)
    return results

def predict_floorplans(num_samples=5, cond_value=0.2, gen_path="models/generator_ep100.pth"):
    gen_model = load_generator(gen_path=gen_path)
    floors = generate_floorplans(gen_model, num_samples, cond_value)
    return floors

if __name__=="__main__":
    os.makedirs("models", exist_ok=True)
    # 利用可能なモデルを表示
    models = get_available_models()
    print("利用可能なモデル:")
    for i, model in enumerate(models):
        print(f"{i}: {model}")
    
    if models:
        # モデルを選択
        idx = int(input("使用するモデルの番号を入力してください: "))
        if 0 <= idx < len(models):
            model_path = os.path.join("models", models[idx])
            # テスト生成
            sample_floors = predict_floorplans(num_samples=3, cond_value=0.3, gen_path=model_path)
            for i, f2d in enumerate(sample_floors):
                print(f"[Generated {i}] shape={f2d.shape}")
                print(f2d)
        else:
            print("無効なモデル番号です")
    else:
        print("利用可能なモデルが見つかりません")
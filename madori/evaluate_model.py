# evaluate_model.py

import numpy as np
import pandas as pd
from scipy import linalg
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3

# 評価するデータファイルのパス
real_data_csv = "floorplan_data.csv"         # 実データ（評価基準となる本物の間取りデータ）
generated_data_csv = "generated_floorplans.csv"  # 生成間取りデータ

# グリッドサイズとクラス数（学習時と同じものを使用）
H, W = 32, 32           # 間取りグリッドの高さ・幅
num_classes = 5         # 部屋カテゴリ数

# 1. FIDスコアの計算
# CSVから実データと生成データを読み込み
real_df = pd.read_csv(real_data_csv)
gen_df = pd.read_csv(generated_data_csv, header=None)
real_array = real_df.values.reshape(-1, H, W)
gen_array = gen_df.values.reshape(-1, H, W)

# InceptionV3モデルを準備（特徴抽出用に全結合層をIdentityに置換）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inception = inception_v3(pretrained=True, transform_input=False).to(device)
inception.fc = torch.nn.Identity()  # 2048次元の特徴ベクトルを直接出力
inception.eval()

# 画像変換: グレースケールの間取りを3チャンネルRGBに拡張しInception入力サイズにリサイズ
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),  # InceptionV3の入力サイズ
    transforms.Grayscale(num_output_channels=3),  # 3チャンネルに変換（各チャンネル同じ値）
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])  # Inceptionの学習時の正規化
])

# 特徴量の抽出関数
def get_inception_features(images_array):
    features_list = []
    for img in images_array:
        img = img.astype(np.uint8)  # ピクセル値を整数型に（カテゴリを疑似的に画素値と扱う）
        img_tensor = transform(img)    # 3x299x299テンソルに変換
        img_tensor = img_tensor.unsqueeze(0).to(device)  # バッチ次元を追加
        with torch.no_grad():
            feat = inception(img_tensor)
        features_list.append(feat.cpu().numpy().reshape(-1))
    features = np.array(features_list)
    return features

# 実データと生成データのInception特徴量を取得
real_features = get_inception_features(real_array)
gen_features  = get_inception_features(gen_array)

# 平均と共分散を計算
mu_real, mu_gen = real_features.mean(axis=0), gen_features.mean(axis=0)
cov_real = np.cov(real_features, rowvar=False)
cov_gen  = np.cov(gen_features,  rowvar=False)
# 共分散行列の平方根を計算
cov_mean = linalg.sqrtm(cov_real.dot(cov_gen))
# 数値誤差で虚数成分が出た場合は実部を使用
if np.iscomplexobj(cov_mean):
    cov_mean = cov_mean.real
# FIDスコアを計算
fid_score = np.sum((mu_real - mu_gen)**2) + np.trace(cov_real + cov_gen - 2 * cov_mean)
print(f"FIDスコア: {fid_score:.4f}")

# 2. 制約適合率の評価
# 例: 必須の部屋カテゴリ（例として1, 2, 3番のカテゴリ）が全て含まれているかチェック
required_classes = [1, 2, 3]  # ドメインに合わせて設定
def satisfies_constraints(layout):
    """間取りレイアウトが全ての必須カテゴリを含むか判定"""
    # layout: 2次元numpy配列 (H, W) カテゴリID
    for cls in required_classes:
        if cls not in layout:
            return False
    return True

# 生成データの各サンプルについて制約をチェック
num_valid = 0
for layout in gen_array:
    if satisfies_constraints(layout):
        num_valid += 1
compliance_rate = num_valid / len(gen_array)
print(f"制約適合率: {compliance_rate*100:.2f}% ({num_valid}/{len(gen_array)})")

# （オプション）実データ側の制約適合率も参考値として計算
num_valid_real = sum(1 for layout in real_array if satisfies_constraints(layout))
real_compliance = num_valid_real / len(real_array)
print(f"(参考)実データの制約適合率: {real_compliance*100:.2f}%")
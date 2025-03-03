import numpy as np
import pandas as pd
from scipy import linalg
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3

# FID計算用CSV(未使用の場合は無視)
real_data_csv = "floorplan_data.csv"
generated_data_csv = "generated_floorplans.csv"

H, W = 32, 32
num_classes = 5

# -------------------------------
# FID計算関連
# -------------------------------
def compute_fid_score(real_array, gen_array):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.fc = torch.nn.Identity()  # 2048次元の特徴ベクトルを直接出力
    inception.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    def get_inception_features(images_array):
        features_list = []
        for img in images_array:
            img = img.astype(np.uint8)
            img_tensor = transform(img)
            img_tensor = img_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                feat = inception(img_tensor)
            features_list.append(feat.cpu().numpy().reshape(-1))
        features = np.array(features_list)
        return features

    real_features = get_inception_features(real_array)
    gen_features  = get_inception_features(gen_array)

    mu_real, mu_gen = real_features.mean(axis=0), gen_features.mean(axis=0)
    cov_real = np.cov(real_features, rowvar=False)
    cov_gen  = np.cov(gen_features,  rowvar=False)

    cov_mean = linalg.sqrtm(cov_real.dot(cov_gen))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    fid_score = np.sum((mu_real - mu_gen)**2) + np.trace(cov_real + cov_gen - 2 * cov_mean)
    return fid_score

# -------------------------------
# 部屋充足チェック
# -------------------------------
def check_required_rooms(generated_layout, required_rooms):
    """
    generated_layout: 2次元配列(H',W'), 各セルが部屋ラベル(str)
    required_rooms: ["l","d","k","t","b"] のように絶対必要な部屋リスト
    戻り値: bool (全て含まれていればTrue)
    """
    found_rooms = set(generated_layout.reshape(-1))
    for r in required_rooms:
        if r not in found_rooms:
            return False
    return True

def check_multiple_or_rooms(generated_layout, or_room_sets):
    """
    generated_layout: 2次元配列
    or_room_sets: 複数の OR 条件をリストで指定
      例: [ ["r","r1"], ["c","c1"] ]
         => (r or r1) かつ (c or c1) が必要
    戻り値: bool
    """
    found = set(generated_layout.reshape(-1))
    for or_rooms in or_room_sets:
        # いずれか一つが含まれていればOK
        if not any(r in found for r in or_rooms):
            return False
    return True

# -------------------------------
# 接続性などの制約
# -------------------------------
def check_connectivity_constraints(generated_layout):
    """
    例: co が全体の5%未満ならNG、などの簡易チェック
    """
    h, w = generated_layout.shape
    total_cells = h * w
    co_count = np.sum(generated_layout == 'co')
    if co_count < total_cells * 0.05:
        return False
    return True

# -------------------------------
# メイン評価関数
# -------------------------------
def evaluate_generated_layouts(gen_array, required_rooms, or_room_sets=None):
    """
    gen_array: list of 2D layouts, 各要素は (H, W) np.array
    required_rooms: ["l","d","k","t","b"] 等
    or_room_sets: OR条件の部屋集合を指定(例: [ ["r","r1"], ["c","c1"] ])
    戻り値: dict {"total_samples", "num_rooms_ok", "num_constraints_ok"}
    """
    results = {
        "total_samples": len(gen_array),
        "num_rooms_ok": 0,
        "num_constraints_ok": 0
    }

    for layout in gen_array:
        # (1) 必須部屋チェック
        if not check_required_rooms(layout, required_rooms):
            pass
        else:
            # 必須部屋が揃ったら OR部屋チェック
            if or_room_sets is not None:
                if check_multiple_or_rooms(layout, or_room_sets):
                    results["num_rooms_ok"] += 1
            else:
                # OR条件なし
                results["num_rooms_ok"] += 1

        # (2) 接続・制約チェック
        if check_connectivity_constraints(layout):
            results["num_constraints_ok"] += 1

    return results

# -------------------------------
# (以下、スクリプト実行時のテスト)
# -------------------------------
if __name__ == "__main__":
    # 例: FIDを計算
    try:
        real_df = pd.read_csv(real_data_csv, header=None)
        gen_df  = pd.read_csv(generated_data_csv, header=None)

        real_array = real_df.values.reshape(-1, H, W)
        gen_array  = gen_df.values.reshape(-1, H, W)

        fid_score_val = compute_fid_score(real_array, gen_array)
        print(f"FIDスコア: {fid_score_val:.4f}")
    except Exception as e:
        print("FIDスコア計算時のエラー:", e)

    # 例: 必須部屋 & ORセットの評価
    required_rooms_ex = ["l","d","k","b","t"]
    or_sets_ex = [ ["r","r1"], ["c","c1"] ]
    # ダミーの生成結果を仮定
    # gen_array_ex = [...]
    # evaluate_results = evaluate_generated_layouts(gen_array_ex, required_rooms_ex, or_sets_ex)
    # print(evaluate_results)
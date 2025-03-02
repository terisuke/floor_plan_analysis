import numpy as np
import pandas as pd
from scipy import linalg
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3

# 任意: 今回は従来のFID計算を残しつつ、部屋充足性や制約チェックを追加
# 下記は元のFID関連
real_data_csv = "floorplan_data.csv"
generated_data_csv = "generated_floorplans.csv"

H, W = 32, 32
num_classes = 5

# もとのFID計算用
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

# ================================
# (1) 部屋の充足率を確認する例
# ================================
def check_required_rooms(generated_layout, required_rooms):
    """
    generated_layout: 2次元配列 (H',W') 各セルが部屋ラベル
    required_rooms: ["L","D","K","t",...] などの必須部屋コードリスト
    戻り値: bool (全て含まれていればTrue)
    """
    found_rooms = set(generated_layout.reshape(-1))
    # '.' や空白は部屋とみなさない
    # すべて必須部屋が found_rooms に含まれているかをチェック
    for r in required_rooms:
        if r not in found_rooms:
            return False
    return True

# ================================
# (2) 接続性などの制約をチェックする例
# ================================
def check_connectivity_constraints(generated_layout):
    """
    例として、部屋が孤立していないか確認。
    ここでは簡易的に、「ある部屋が存在するなら、その周囲に'.'または同じ部屋ラベル以外でも通路('co')があるか」を見る。
    より厳密には BFS/DFS で部屋全体を探査し、孤立していないかなどを確かめる。
    戻り値: bool (OKならTrue)
    """
    # ここでは省略し、必須部屋があってかつ半分以上のセルが何らかの通路か連結マスであればOK等の簡単な例:
    h, w = generated_layout.shape
    total_cells = h*w
    # co(廊下)セル数が極端に少ないとNG、としてみる
    co_count = np.sum(generated_layout == 'co')
    if co_count < total_cells*0.05:
        return False
    return True

# ================================
# メイン評価関数例
# ================================
def evaluate_generated_layouts(gen_array, required_rooms):
    """
    gen_array: [N, H', W'] 生成レイアウトのバッチ
    required_rooms: 必須部屋リスト e.g. ["L", "D","K","B","t","e","H"] など
    戻り値: dict に各種指標をまとめる
    """
    results = {
        "total_samples": len(gen_array),
        "num_rooms_ok": 0,
        "num_constraints_ok": 0
    }

    for layout in gen_array:
        # layoutは2次元の場合: shape=(H', W')
        # np.unique()等でラベル確認
        # (1) 必須部屋確認
        if check_required_rooms(layout, required_rooms):
            results["num_rooms_ok"] += 1
        # (2) 接続/制約確認
        if check_connectivity_constraints(layout):
            results["num_constraints_ok"] += 1

    return results


if __name__ == "__main__":
    # 例: FID計算
    try:
        real_df = pd.read_csv(real_data_csv, header=None)
        gen_df  = pd.read_csv(generated_data_csv, header=None)

        real_array = real_df.values.reshape(-1, H, W)
        gen_array  = gen_df.values.reshape(-1, H, W)

        fid_score_val = compute_fid_score(real_array, gen_array)
        print(f"FIDスコア: {fid_score_val:.4f}")
    except Exception as e:
        print("FIDスコア計算時のエラー:", e)

    # 例: 必須の部屋が ["L","D","K","B","t","H","e"] だとして評価
    # CSVでの文字ラベルに合わせてる場合 (string)
    required_rooms = ["L","D","K","B","t","H","e"]

    # ここではダミー例: gen_arrayを使って評価
    # gen_arrayが int型なら decodeが必要かもしれません
    # いったんstring想定として
    if len(gen_array.shape) == 3:
        # N,H,W
        # shapeを (N,H,W) -> list of (H,W)
        gen_list = []
        for i in range(gen_array.shape[0]):
            # 文字列ラベル想定であれば astype(str) する必要がある
            # ここでは簡易的にchr()で変換するなど
            layout_2d = gen_array[i]
            # もとのラベルがintならマップが必要
            # ここでは省略
            gen_list.append(layout_2d)
        # 評価
        eval_results = evaluate_generated_layouts(gen_list, required_rooms)
        print(f"部屋充足OK: {eval_results['num_rooms_ok']} / {eval_results['total_samples']}")
        print(f"制約OK   : {eval_results['num_constraints_ok']} / {eval_results['total_samples']}")
    else:
        print("gen_array shape が想定外です。")
"""
Streamlitを用いて、ユーザが間取りサイズや部屋数を指定し、
cGANから新規間取りを生成するデモ用アプリ。
"""
import streamlit as st
import torch
import numpy as np
from madori.gan import Generator
from madori.predict_model import get_available_models, NOISE_DIM, COND_DIM, OUT_CHANNELS
from PIL import Image
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_cgan_generator(ckpt_path, noise_dim=NOISE_DIM, cond_dim=COND_DIM, out_channels=23):
    netG = Generator(
        noise_dim=noise_dim, 
        cond_dim=cond_dim, 
        out_channels=out_channels,
        base_channels=96  # チェックポイントと一致するように修正（96->192->384->768の順）
    )
    netG.load_state_dict(torch.load(ckpt_path, map_location=device))
    netG.eval().to(device)
    return netG

def crop_layout(layout, height, width):
    """生成された間取り図を指定サイズにクロップ"""
    return layout[:height, :width]

def auto_trim_layout(layout_2d):
    """
    layout_2d: 2次元 numpy配列 (shape=(H,W))
    '.' (ドット) は空白とみなし、上下左右にある空白行・列を切り詰める。
    完全に空白な場合はそのまま返す。
    """
    import numpy as np
    
    # True/Falseマスクで、各行または列に少なくとも1つの空白以外セルがあるか判定
    row_has_room = np.any(layout_2d != ".", axis=1)
    col_has_room = np.any(layout_2d != ".", axis=0)
    
    # 全て空白なら、そのまま返す
    if not np.any(row_has_room) or not np.any(col_has_room):
        return layout_2d
    
    # row_has_roomがTrueの行だけ取り出し → その中で col_has_roomがTrueの列だけ取り出す
    trimmed = layout_2d[row_has_room][:, col_has_room]
    return trimmed

def main():
    st.title("Floor Plan Generator (cGAN)")

    # 利用可能なモデルを取得
    available_models = get_available_models()
    
    if not available_models:
        st.error("利用可能なモデルが見つかりません。'models'ディレクトリにgenerator_*.pthファイルが必要です。")
        return

    # サイドバーでモデル選択
    selected_model = st.sidebar.selectbox(
        "生成モデルを選択",
        available_models,
        format_func=lambda x: f"モデル: {x}"
    )
    
    gen_ckpt = os.path.join("models", selected_model)
    
    st.sidebar.write("ノイズ次元:", NOISE_DIM)
    st.sidebar.write("条件次元:", COND_DIM)

    # 自動トリミングのチェックボックス
    auto_trim = st.sidebar.checkbox("自動トリミングを行う", value=False, 
                                  help="生成後に外周の空白('.'のみの行・列)を切り詰めます。")

    # レイアウトサイズの入力
    col1, col2 = st.columns(2)
    with col1:
        height = st.slider("縦のマス数", min_value=5, max_value=64, value=32)
    with col2:
        width = st.slider("横のマス数", min_value=5, max_value=64, value=32)

    # 部屋の割合を条件として入力
    st.write("### 部屋の設定")
    cond_val = st.slider("部屋の占有率 (0~1)", min_value=0.0, max_value=1.0, value=0.3, step=0.05,
                        help="部屋が占める面積の割合。大きいほど部屋が多く生成されます。")
    
    num_gen = st.slider("生成する間取り数", min_value=1, max_value=5, value=3)

    if st.button("間取りを生成"):
        st.info("Generatorを読み込み中...")
        netG = load_cgan_generator(gen_ckpt)
        st.success("Generator読み込み完了")

        st.info("間取りを生成中...")
        results = []
        with torch.no_grad():
            for i in range(num_gen):
                z = torch.randn(1, NOISE_DIM, device=device)
                cond = torch.tensor([[cond_val]], dtype=torch.float, device=device)
                fake_logits = netG(z, cond)
                pred = fake_logits.argmax(dim=1).squeeze(0).cpu().numpy()
                
                # まずは指定サイズ (height, width) にクロップ
                cropped_pred = crop_layout(pred, height, width)

                # もし「自動トリミング」がONなら、さらに外周の空白を除去
                if auto_trim:
                    cropped_pred = auto_trim_layout(cropped_pred)
                
                results.append(cropped_pred)

        st.success("生成完了!")
        
        # 可視化
        for i, arr_2d in enumerate(results):
            st.subheader(f"生成された間取り #{i+1}")
            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(arr_2d, cmap='tab20')
            ax.set_title(f"間取り (サイズ: {arr_2d.shape[0]}×{arr_2d.shape[1]}マス)")
            ax.grid(True)
            plt.colorbar(im)
            st.pyplot(fig)
            plt.close()

            # 間取りの統計情報
            unique, counts = np.unique(arr_2d, return_counts=True)
            room_stats = dict(zip(unique, counts))
            st.write("部屋の統計:")
            st.write(f"- 総マス数: {arr_2d.size}")
            st.write(f"- 部屋の種類数: {len(unique)}")
            st.write(f"- 各部屋の出現回数: {room_stats}")

if __name__=="__main__":
    main()
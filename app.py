
import os
import glob
from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt

from Model.Model import PINN
from dataloader.dataloader import DF
from utils.util import eval_metrix

# ==============================================================================
# CẤU HÌNH TRANG 
# ==============================================================================
st.set_page_config(page_title="PINN4SOH - Dự đoán SOH pin", page_icon=" ", layout="wide")

# --- CSS tuỳ chỉnh: bo góc, màu pastel, thẻ (card) mềm mại ------------------
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #fdfbff 0%, #f6f9ff 100%);
        }
        h1, h2, h3 { color: #4a3f8a; }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #eae6ff;
            border-radius: 16px;
            padding: 10px 6px;
            box-shadow: 0 2px 8px rgba(120, 100, 220, 0.08);
        }
        .info-card {
            background: #f4f1ff;
            border: 1px solid #e2dbff;
            border-radius: 16px;
            padding: 16px 20px;
            margin-bottom: 10px;
        }
        .stButton>button {
            border-radius: 999px;
            font-weight: 600;
        }
        section[data-testid="stSidebar"] {
            background: #faf7ff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("PINN4SOH — Dự đoán sức khỏe pin (SOH)")
st.caption("Mô hình học sâu kết hợp vật lý (Physics-Informed Neural Network) để dự đoán SOH pin từ dữ liệu sạc/xả thật.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Thông tin mặc định cho từng bộ dữ liệu: dung lượng danh định (Ah)
# và cấu hình mạng nhánh F (phải khớp với lúc huấn luyện).
DATASET_INFO = {
    "XJTU": {"nominal_capacity": 2.0, "F_layers_num": 3, "F_hidden_dim": 60},
    "HUST": {"nominal_capacity": 1.1, "F_layers_num": 3, "F_hidden_dim": 60},
    "MIT":  {"nominal_capacity": 1.1, "F_layers_num": 3, "F_hidden_dim": 60},
    "TJU":  {"nominal_capacity": 3.5, "F_layers_num": 3, "F_hidden_dim": 60},
}

# Giải thích các chỉ số đánh giá để hiển thị cạnh kết quả
METRIC_EXPLAIN = {
    "MAE":  "Sai số tuyệt đối trung bình. Trung bình cộng độ lệch (không tính dấu) giữa giá trị dự đoán và thực tế. Càng gần 0 càng tốt.",
    "RMSE": "Căn bậc hai sai số bình phương trung bình. Giống MAE nhưng phạt nặng hơn với các sai số lớn. Càng nhỏ càng tốt.",
    "MAPE": "Sai số phần trăm tuyệt đối trung bình. Cho biết mô hình sai lệch trung bình bao nhiêu % so với giá trị thật.",
    "MSE":  "Sai số bình phương trung bình. Bình phương độ lệch trước khi lấy trung bình, dùng nội bộ để huấn luyện mô hình.",
}


# ==============================================================================
# CÁC HÀM TIỆN ÍCH
# ==============================================================================
def find_checkpoints():
    """Quét toàn bộ project để tìm các file checkpoint (model.pth hoặc *.pth)
    đã được lưu bởi quá trình huấn luyện (PINN.Train)."""
    paths = glob.glob("**/model.pth", recursive=True)
    paths += glob.glob("**/*.pth", recursive=True)
    paths = sorted(set(paths))
    return paths


def build_args(dataset_name, normalization_method):
    """Tạo đối tượng args tối thiểu để khởi tạo PINN(args) phục vụ INFERENCE.

    Lưu ý:
    - Các tham số về scheduler/loss không ảnh hưởng đến kết quả dự đoán,
      nhưng PINN.__init__ vẫn cần chúng để khởi tạo mà không bị lỗi.
    - F_layers_num / F_hidden_dim BẮT BUỘC phải khớp với lúc huấn luyện
      (mặc định là 3 / 60 trong tất cả các file main_*.py đã tham khảo).
    """
    info = DATASET_INFO[dataset_name]
    tmp_folder = os.path.join("streamlit_tmp", dataset_name)
    os.makedirs(tmp_folder, exist_ok=True)

    return SimpleNamespace(
        data=dataset_name,
        batch_size=256,
        normalization_method=normalization_method,
        epochs=1,
        early_stop=None,
        warmup_epochs=1,
        warmup_lr=1e-3,
        lr=1e-3,
        final_lr=1e-4,
        lr_F=1e-3,
        F_layers_num=info["F_layers_num"],
        F_hidden_dim=info["F_hidden_dim"],
        alpha=1.0,
        beta=1.0,
        log_dir="inference_log.txt",
        save_folder=tmp_folder,
    )


@st.cache_resource(show_spinner=False)
def load_pinn(dataset_name, normalization_method, checkpoint_path):
    """Tải mô hình PINN từ checkpoint và chuyển sang chế độ đánh giá (eval)."""
    args = build_args(dataset_name, normalization_method)
    pinn = PINN(args)
    pinn.load_model(checkpoint_path)
    pinn.eval()
    return pinn


def predict_from_csv(pinn, csv_path, nominal_capacity, normalization_method):
    """Đọc một file CSV theo đúng pipeline lúc huấn luyện (DF.read_one_csv),
    sau đó dự đoán capacity/SOH cho từng dòng bằng hàm solution_u."""
    args = SimpleNamespace(
        normalization_method=normalization_method,
        log_dir=None,
        save_folder=None,
    )
    dfproc = DF(args)
    df = dfproc.read_one_csv(csv_path, nominal_capacity=nominal_capacity)

    x = df.iloc[:, :-1].values.astype(np.float32)
    y_true = df.iloc[:, -1].values.astype(np.float32)

    x_tensor = torch.from_numpy(x).float().to(DEVICE)
    with torch.no_grad():
        y_pred = pinn.predict(x_tensor).cpu().numpy().flatten()

    return y_true, y_pred, df


# ==============================================================================
# HỘP GIẢI THÍCH KHÁI NIỆM (để người mới cũng hiểu ứng dụng đang làm gì)
# ==============================================================================
with st.expander("💡 SOH là gì? Vì sao cần dự đoán?", expanded=False):
    st.markdown(
        """
        **SOH (State of Health)** là chỉ số thể hiện "sức khỏe" hiện tại của pin,
        tính bằng tỉ lệ giữa dung lượng hiện tại và dung lượng lúc pin còn mới.

        - SOH càng gần **100%** → pin còn tốt, gần như mới.
        - SOH càng thấp → pin đã xuống cấp, dung lượng lưu trữ giảm dần theo thời gian sử dụng.

        Mô hình **PINN (Physics-Informed Neural Network)** ở đây học từ dữ liệu sạc/xả thật
        kết hợp với các ràng buộc vật lý, giúp dự đoán SOH chính xác hơn ngay cả khi
        dữ liệu quan sát được còn hạn chế.
        """
    )

# ==============================================================================
# THANH BÊN (SIDEBAR) - CHỌN MÔ HÌNH VÀ DỮ LIỆU
# ==============================================================================
st.sidebar.header("1️ Chọn mô hình đã huấn luyện")

dataset_name = st.sidebar.selectbox(
    "🔹 Bộ dữ liệu",
    list(DATASET_INFO.keys()),
    help="Bộ dữ liệu pin dùng để huấn luyện mô hình tương ứng (XJTU, HUST, MIT, TJU).",
)
normalization_method = st.sidebar.selectbox(
    "🔹 Phương pháp chuẩn hóa",
    ["min-max", "z-score"],
    help="Cách đưa dữ liệu về cùng một thang đo trước khi đưa vào mô hình. "
         "Bắt buộc phải chọn giống lúc huấn luyện, nếu không kết quả sẽ sai.",
)

checkpoints = find_checkpoints()
checkpoints_filtered = [c for c in checkpoints if dataset_name.lower() in c.lower()]
show_list = checkpoints_filtered if checkpoints_filtered else checkpoints

if not show_list:
    st.sidebar.error(
        " Không tìm thấy file model.pth / *.pth nào trong project.\n\n"
        "Hãy chắc chắn bạn đã huấn luyện xong (ví dụ chạy main_XJTU.py) và "
        "checkpoint đã được lưu trong thư mục results/..."
    )
    st.stop()

checkpoint_path = st.sidebar.selectbox(
    "🔹 Checkpoint (model.pth)",
    show_list,
    help="File trọng số (weights) của mô hình đã huấn luyện xong, dùng để nạp vào và dự đoán.",
)

st.sidebar.header("2️ Chọn dữ liệu để dự đoán")
default_cap = DATASET_INFO[dataset_name]["nominal_capacity"]
nominal_capacity = st.sidebar.number_input(
    "🔹 Dung lượng danh định (Ah)",
    value=float(default_cap),
    step=0.1,
    help="Dung lượng pin lúc còn mới (100% SOH). Dùng làm mốc để quy đổi capacity → SOH.",
)

data_source = st.sidebar.radio(
    "🔹 Nguồn dữ liệu",
    ["Chọn file có sẵn trong project", "Tải lên file CSV mới"],
)

csv_path = None
uploaded_tmp_path = None

if data_source == "Chọn file có sẵn trong project":
    all_csv = glob.glob("data/**/*.csv", recursive=True)
    dataset_csv = [f for f in all_csv if dataset_name.lower() in f.lower()] or all_csv

    if not dataset_csv:
        st.sidebar.warning(" Không tìm thấy file CSV nào trong thư mục data/.")
    else:
        csv_path = st.sidebar.selectbox(" Chọn file pin (.csv)", dataset_csv)
else:
    uploaded = st.sidebar.file_uploader(
        " Tải lên file CSV (đúng định dạng như dữ liệu gốc)", type=["csv"]
    )
    if uploaded is not None:
        os.makedirs("streamlit_tmp", exist_ok=True)
        uploaded_tmp_path = os.path.join("streamlit_tmp", uploaded.name)
        with open(uploaded_tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        csv_path = uploaded_tmp_path

run_btn = st.sidebar.button(" Dự đoán ", type="primary", use_container_width=True)

# ==============================================================================
# NỘI DUNG CHÍNH
# ==============================================================================
st.markdown(
    f"""
    <div class="info-card">
     <b>Mô hình:</b> <code>{checkpoint_path}</code> &nbsp;·&nbsp;
     <b>Bộ dữ liệu:</b> <code>{dataset_name}</code> &nbsp;·&nbsp;
     <b>Chuẩn hóa:</b> <code>{normalization_method}</code>
    </div>
    """,
    unsafe_allow_html=True,
)

if run_btn:
    if csv_path is None:
        st.error(" Bạn chưa chọn hoặc tải lên file CSV nào.")
        st.stop()

    with st.spinner(" Đang tải mô hình..."):
        try:
            pinn = load_pinn(dataset_name, normalization_method, checkpoint_path)
        except Exception as e:
            st.error(f" Lỗi khi tải checkpoint: {e}")
            st.stop()

    with st.spinner(" Đang dự đoán..."):
        try:
            y_true, y_pred, df = predict_from_csv(
                pinn, csv_path, nominal_capacity, normalization_method
            )
        except Exception as e:
            st.error(f" Lỗi khi xử lý file CSV / dự đoán: {e}")
            st.exception(e)
            st.stop()

    MAE, MAPE, MSE, RMSE = eval_metrix(y_pred, y_true)

    tab1, tab2, tab3 = st.tabs([" Chỉ số đánh giá", " Biểu đồ", " Bảng dữ liệu"])

    # --- TAB 1: Chỉ số đánh giá --------------------------------------------
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"{MAE:.5f}")
        c2.metric("RMSE", f"{RMSE:.5f}")
        c3.metric("MAPE", f"{MAPE:.5f}")
        c4.metric("MSE", f"{MSE:.6f}")

        with st.expander(" Các chỉ số này nghĩa là gì?"):
            for name, desc in METRIC_EXPLAIN.items():
                st.markdown(f"- **{name}**: {desc}")

    # --- TAB 2: Biểu đồ so sánh ---------------------------------------------
    with tab2:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(y_true, label="Thực tế (true SOH)", linewidth=2, color="#7c6bd6")
        ax.plot(y_pred, label="Dự đoán (predicted SOH)", linestyle="--", color="#ff9ecb")
        ax.set_xlabel("Chu kỳ (cycle index trong file)")
        ax.set_ylabel("SOH / Capacity (đã chuẩn hóa)")
        ax.set_title(f"Dự đoán SOH — {os.path.basename(csv_path)}")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        st.caption(" Đường tím là giá trị thật, đường hồng nét đứt là giá trị mô hình dự đoán. "
                   "Hai đường càng sát nhau, mô hình dự đoán càng chính xác.")

    # --- TAB 3: Bảng kết quả + tải về ---------------------------------------
    with tab3:
        result_df = pd.DataFrame({"SOH_thực_tế": y_true, "SOH_dự_đoán": y_pred})
        st.dataframe(result_df, use_container_width=True)

        csv_download = result_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇ Tải kết quả (.csv)",
            csv_download,
            file_name=f"du_doan_{os.path.basename(csv_path)}",
            mime="text/csv",
        )
else:
    st.info("Chọn mô hình và dữ liệu ở thanh bên trái, rồi bấm ** Dự đoán **.")

import os
import io
import json
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. 기본 설정
# =========================================================

st.set_page_config(
    page_title="Industrial AI | Tool Wear Monitor",
    page_icon="⚙️",
    layout="wide"
)

DB_PATH = "industrial_records.db"
DEFAULT_DATA_PATH = "ai4i2020.csv"

TARGET = "Tool wear [min]"

ALL_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]"
]

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


st.markdown("""
<style>
/* =====================================
   GLOBAL THEME
===================================== */

:root {
    --bg: #0b1120;
    --sidebar: #0e1728;
    --panel: #141f33;
    --panel-hover: #1a2940;
    --border: #263750;
    --accent: #38bdf8;
    --accent-dark: #0284c7;
    --text: #f8fafc;
    --muted: #cbd5e1;
}

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: "Pretendard", "Noto Sans KR", sans-serif;
}

/* 모든 기본 텍스트 흰색 계열 */

html, body, [class*="css"],
p, span, label, li, strong, small,
div, h1, h2, h3, h4, h5, h6 {
    color: var(--text);
}

/* 설명 및 보조 텍스트 */

.stCaption, [data-testid="stCaptionContainer"],
small {
    color: var(--muted) !important;
}

/* =====================================
   MAIN LAYOUT
===================================== */

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

/* =====================================
   SIDEBAR
===================================== */

[data-testid="stSidebar"] {
    background: var(--sidebar);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

[data-testid="stSidebar"] h1 {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: 1px;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    background: transparent;
    border-radius: 8px;
    padding: 9px 12px;
    transition: 0.2s;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #1c2d45;
}

/* =====================================
   HEADINGS
===================================== */

h1 {
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.8px;
    color: #ffffff !important;
}

h2 {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #f8fafc !important;
}

h3 {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: #e2e8f0 !important;
}

/* =====================================
   CARDS AND METRICS
===================================== */

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #17253a, #111d30);
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 18px 20px;
    box-shadow: 0 5px 18px rgba(0,0,0,0.16);
}

[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
    font-size: 0.9rem;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 750;
}

[data-testid="stMetricDelta"] {
    color: #7dd3fc !important;
}

/* =====================================
   BUTTONS
===================================== */

.stButton > button,
.stDownloadButton > button {
    background: #147da5;
    color: #ffffff !important;
    border: 1px solid #2496bd;
    border-radius: 9px;
    padding: 0.55rem 1rem;
    font-weight: 650;
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #1598c5;
    border-color: #38bdf8;
    color: #ffffff !important;
    transform: translateY(-1px);
}

.stButton > button p,
.stDownloadButton > button p {
    color: #ffffff !important;
}

/* =====================================
   INPUTS
===================================== */

.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
[data-baseweb="select"] *,
[data-baseweb="input"] *,
[data-baseweb="textarea"] * {
    background-color: #101c2e !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border-color: #344861 !important;
}

input::placeholder,
textarea::placeholder {
    color: #94a3b8 !important;
    -webkit-text-fill-color: #94a3b8 !important;
}

[data-baseweb="select"] > div {
    background-color: #101c2e !important;
    border-color: #344861 !important;
}

[data-baseweb="popover"],
[data-baseweb="menu"],
[data-baseweb="menu"] li {
    background-color: #17253a !important;
    color: #ffffff !important;
}

[data-baseweb="menu"] li:hover {
    background-color: #263b56 !important;
}

/* =====================================
   SLIDERS AND CHECKBOXES
===================================== */

.stSlider label,
.stCheckbox label,
.stRadio label,
.stMultiSelect label,
.stSelectbox label,
.stNumberInput label,
.stTextInput label,
.stTextArea label {
    color: #f1f5f9 !important;
}

[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: #cbd5e1 !important;
}

/* =====================================
   DATAFRAME / TABLE
===================================== */

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background: #121e31;
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}

[data-testid="stDataFrame"] *,
[data-testid="stTable"] * {
    color: #f8fafc !important;
}

/* =====================================
   TABS
===================================== */

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    color: #cbd5e1 !important;
    background: transparent;
}

.stTabs [aria-selected="true"] {
    color: #7dd3fc !important;
    border-bottom: 2px solid #38bdf8 !important;
}

/* =====================================
   ALERTS / EXPANDERS
===================================== */

[data-testid="stAlert"] {
    border-radius: 10px;
    border: 1px solid var(--border);
}

[data-testid="stExpander"] {
    background: #121e31;
    border: 1px solid var(--border);
    border-radius: 10px;
}

/* =====================================
   DIVIDERS / SCROLLBAR
===================================== */

hr {
    border-color: #263750 !important;
}

::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #0b1120;
}

::-webkit-scrollbar-thumb {
    background: #344861;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)



# =========================================================
# 2. 데이터베이스
# =========================================================

def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                prediction REAL,
                input_data TEXT,
                source TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                experiment_name TEXT,
                features TEXT,
                test_size REAL,
                mae REAL,
                mse REAL,
                rmse REAL,
                r2 REAL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS maintenance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                equipment TEXT,
                maintenance_type TEXT,
                wear_value REAL,
                description TEXT
            )
        """)


def save_prediction(prediction, input_data, source="직접 예측"):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO predictions
            (created_at, prediction, input_data, source)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            float(prediction),
            json.dumps(input_data, ensure_ascii=False),
            source
        ))


def load_predictions():
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY id DESC",
            conn
        )


def save_experiment(name, features, test_size, mae, mse, rmse, r2):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO experiments
            (created_at, experiment_name, features, test_size,
             mae, mse, rmse, r2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            name,
            json.dumps(features, ensure_ascii=False),
            float(test_size),
            float(mae),
            float(mse),
            float(rmse),
            float(r2)
        ))


def load_experiments():
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM experiments ORDER BY id DESC",
            conn
        )


def save_maintenance(equipment, maintenance_type, wear_value, description):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO maintenance
            (created_at, equipment, maintenance_type, wear_value, description)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            equipment,
            maintenance_type,
            float(wear_value),
            description
        ))


def load_maintenance():
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM maintenance ORDER BY id DESC",
            conn
        )


def delete_record(table, record_id):
    allowed_tables = {"predictions", "experiments", "maintenance"}
    if table not in allowed_tables:
        return

    with get_connection() as conn:
        conn.execute(
            f"DELETE FROM {table} WHERE id = ?",
            (int(record_id),)
        )


def update_maintenance(record_id, equipment, maintenance_type,
                       wear_value, description):
    with get_connection() as conn:
        conn.execute("""
            UPDATE maintenance
            SET equipment = ?, maintenance_type = ?,
                wear_value = ?, description = ?
            WHERE id = ?
        """, (
            equipment,
            maintenance_type,
            float(wear_value),
            description,
            int(record_id)
        ))


initialize_database()


# =========================================================
# 3. 세션 상태
# =========================================================

defaults = {
    "df": None,
    "model": None,
    "features": [],
    "evaluation": None,
    "test_data": None,
    "prediction_history": [],
    "current_experiment": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 4. 데이터 처리 및 모델 함수
# =========================================================

def clean_data(data):
    data = data.copy()

    for col in NUMERIC_FEATURES:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "Type" in data.columns:
        data["Type"] = data["Type"].astype("string").str.strip()
        data["Type"] = data["Type"].replace("", pd.NA)

    return data


def make_pipeline(data, features):
    categorical = [
        col for col in features
        if (
            col in data.columns
            and (
                pd.api.types.is_object_dtype(data[col])
                or isinstance(data[col].dtype, pd.CategoricalDtype)
                or pd.api.types.is_string_dtype(data[col])
            )
        )
    ]

    numerical = [col for col in features if col not in categorical]

    transformers = []

    if numerical:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numerical)
        )

    if categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical)
        )

    preprocessor = ColumnTransformer(transformers)

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    return model_pipeline


def train_model(data, features, test_size):
    required = features + [TARGET]
    work = data[required].copy()

    work[TARGET] = pd.to_numeric(work[TARGET], errors="coerce")
    work = work.dropna(subset=[TARGET])

    if len(work) < 5:
        raise ValueError("학습에 사용할 수 있는 데이터가 너무 적습니다.")

    X = work[features]
    y = work[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42
    )

    model = make_pipeline(work, features)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    result = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

    test_result = pd.DataFrame({
        "실제 마모 시간": y_test.values,
        "예측 마모 시간": predictions,
        "잔차": y_test.values - predictions
    })

    return model, result, test_result


def reset_model():
    st.session_state.model = None
    st.session_state.features = []
    st.session_state.evaluation = None
    st.session_state.test_data = None
    st.session_state.current_experiment = None


def load_new_data(data):
    st.session_state.df = clean_data(data)
    reset_model()


def get_dataframe():
    if st.session_state.df is None:
        return None
    return st.session_state.df.copy()


def show_metric(label, value, fmt=".3f"):
    st.metric(label, format(value, fmt))


# =========================================================
# 5. 데이터 불러오기
# =========================================================

if st.session_state.df is None:
    if os.path.exists(DEFAULT_DATA_PATH):
        try:
            default_df = pd.read_csv(DEFAULT_DATA_PATH)
            load_new_data(default_df)
        except Exception:
            pass


# =========================================================
# 6. 사이드바
# =========================================================

st.sidebar.title("⚙️ INDUSTRIAL AI")
st.sidebar.caption("TOOL WEAR MONITORING SYSTEM")
st.sidebar.divider()

menu = st.sidebar.radio(
    "시스템 메뉴",
    [
        "🏠 메인 대시보드",
        "📁 데이터 관리",
        "📊 데이터 분석",
        "🧠 AI 모델 실험실",
        "⚙️ 공구 마모 예측",
        "🔧 설비 및 정비 관리",
        "📄 분석 보고서"
    ]
)

st.sidebar.divider()
st.sidebar.caption("AI4I 2020 Predictive Maintenance Dataset")
st.sidebar.caption("Linear Regression | Scikit-learn")


# =========================================================
# 7. 메인 대시보드
# =========================================================

if menu == "🏠 메인 대시보드":

    st.title("산업 설비 모니터링 대시보드")
    st.caption("설비 작동 데이터 기반 공구 마모 예측 및 정비 의사결정 지원")

    df = get_dataframe()

    if df is None:
        st.warning("데이터가 아직 등록되지 않았습니다. 데이터 관리 메뉴에서 CSV를 업로드하세요.")
    else:
        predictions = load_predictions()
        maintenance = load_maintenance()
        experiments = load_experiments()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("전체 데이터", f"{len(df):,}개")
        c2.metric("데이터 변수", f"{len(df.columns)}개")
        c3.metric("예측 기록", f"{len(predictions):,}건")
        c4.metric("정비 기록", f"{len(maintenance):,}건")

        st.divider()

        left, right = st.columns(2)

        with left:
            st.subheader("공구 마모 시간 분포")
            if TARGET in df.columns:
                fig = px.histogram(
                    df,
                    x=TARGET,
                    nbins=30,
                    title="Tool Wear Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("제품 유형별 평균 마모")
            if "Type" in df.columns and TARGET in df.columns:
                type_mean = (
                    df.groupby("Type", observed=True)[TARGET]
                    .mean()
                    .reset_index()
                )

                fig = px.bar(
                    type_mean,
                    x="Type",
                    y=TARGET,
                    color="Type",
                    title="Average Tool Wear by Type"
                )
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("최근 예측 기록")

        if not predictions.empty:
            st.dataframe(
                predictions.head(5),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("아직 예측 기록이 없습니다.")

        st.subheader("시스템 상태")

        status1, status2, status3 = st.columns(3)

        status1.success("데이터 로드 완료")
        status2.success("SQLite 기록 저장 활성화")

        if st.session_state.model is not None:
            status3.success("AI 모델 학습 완료")
        else:
            status3.warning("AI 모델 학습 필요")


# =========================================================
# 8. 데이터 관리
# =========================================================

elif menu == "📁 데이터 관리":

    st.title("데이터 관리")
    st.caption("CSV 데이터 업로드, 결측치 확인, 이상치 탐지 및 데이터 필터링")

    uploaded_file = st.file_uploader(
        "CSV 파일 업로드",
        type=["csv"]
    )

    col1, col2 = st.columns(2)

    with col1:
        if uploaded_file is not None:
            if st.button("업로드 데이터 적용", use_container_width=True):
                try:
                    uploaded_df = pd.read_csv(uploaded_file)
                    load_new_data(uploaded_df)
                    st.success("데이터를 성공적으로 적용했습니다.")
                    st.rerun()
                except Exception as e:
                    st.error(f"CSV 처리 오류: {e}")

    with col2:
        if st.button("기본 AI4I 데이터 불러오기", use_container_width=True):
            if os.path.exists(DEFAULT_DATA_PATH):
                try:
                    load_new_data(pd.read_csv(DEFAULT_DATA_PATH))
                    st.success("기본 데이터가 적용되었습니다.")
                    st.rerun()
                except Exception as e:
                    st.error(f"데이터 로드 오류: {e}")
            else:
                st.error("프로젝트 폴더에서 ai4i2020.csv 파일을 찾을 수 없습니다.")

    df = get_dataframe()

    if df is not None:

        st.divider()
        st.subheader("데이터 미리보기")

        c1, c2, c3 = st.columns(3)
        c1.metric("행 개수", f"{len(df):,}")
        c2.metric("열 개수", f"{len(df.columns)}")
        c3.metric("중복 행", f"{df.duplicated().sum():,}")

        st.dataframe(df.head(100), use_container_width=True)

        st.subheader("데이터 유형")
        dtype_df = pd.DataFrame({
            "변수": df.columns,
            "데이터 유형": df.dtypes.astype(str).values,
            "결측치": df.isnull().sum().values
        })

        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("결측치 분석 및 처리")

        missing = df.isnull().sum()
        missing_df = pd.DataFrame({
            "변수": missing.index,
            "결측치 개수": missing.values
        })

        st.dataframe(missing_df, use_container_width=True, hide_index=True)

        if df.isnull().sum().sum() > 0:
            if st.button("결측치가 포함된 행 제거"):
                cleaned = df.dropna()
                load_new_data(cleaned)
                st.success(f"{len(df) - len(cleaned)}개 행을 제거했습니다.")
                st.rerun()
        else:
            st.success("결측치가 없습니다.")

        if df.duplicated().sum() > 0:
            if st.button("중복 행 제거"):
                cleaned = df.drop_duplicates()
                load_new_data(cleaned)
                st.success("중복 행을 제거했습니다.")
                st.rerun()

        st.divider()
        st.subheader("이상치 탐지 - IQR 방식")

        outlier_col = st.selectbox(
            "이상치를 확인할 변수",
            [c for c in NUMERIC_FEATURES if c in df.columns]
        )

        if outlier_col:
            q1 = df[outlier_col].quantile(0.25)
            q3 = df[outlier_col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outlier_mask = (
                (df[outlier_col] < lower)
                | (df[outlier_col] > upper)
            )

            st.write(f"하한 기준: {lower:.3f}")
            st.write(f"상한 기준: {upper:.3f}")
            st.write(f"탐지된 이상치: {int(outlier_mask.sum())}개")

            if st.button("선택 변수의 이상치 행 제거"):
                cleaned = df.loc[~outlier_mask].copy()
                load_new_data(cleaned)
                st.success("이상치 행을 제거했습니다.")
                st.rerun()

        st.divider()
        st.subheader("데이터 필터링")

        filtered = df.copy()

        if "Type" in filtered.columns:
            types = sorted(filtered["Type"].dropna().astype(str).unique())
            selected_types = st.multiselect(
                "제품 유형",
                types,
                default=types
            )

            filtered = filtered[
                filtered["Type"].astype(str).isin(selected_types)
            ]

        available_numeric = [
            c for c in NUMERIC_FEATURES
            if c in filtered.columns and filtered[c].notna().any()
        ]

        for col in available_numeric:
            if filtered.empty:
                st.warning("현재 조건에 해당하는 데이터가 없습니다.")
                break

            min_value = float(filtered[col].min())
            max_value = float(filtered[col].max())

            if min_value < max_value:
                selected_range = st.slider(
                    col,
                    min_value=min_value,
                    max_value=max_value,
                    value=(min_value, max_value)
                )

                filtered = filtered[
                    filtered[col].between(
                        selected_range[0],
                        selected_range[1]
                    )
                ]

        st.write(f"필터링 결과: {len(filtered):,}개 행")
        st.dataframe(filtered, use_container_width=True)

        st.download_button(
            "필터링 데이터 CSV 다운로드",
            data=filtered.to_csv(index=False).encode("utf-8-sig"),
            file_name="filtered_data.csv",
            mime="text/csv"
        )


# =========================================================
# 9. 데이터 분석
# =========================================================

elif menu == "📊 데이터 분석":

    st.title("데이터 분석")
    st.caption("공구 마모와 설비 작동 변수 사이의 관계 분석")

    df = get_dataframe()

    if df is None:
        st.warning("먼저 데이터를 불러오세요.")
    else:
        numeric_df = df.select_dtypes(include=np.number)

        st.subheader("상관관계 분석")

        corr = numeric_df.corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title="Correlation Heatmap"
        )

        st.plotly_chart(fig, use_container_width=True)

        if TARGET in corr.columns:
            st.subheader("공구 마모 시간과의 상관관계")

            target_corr = (
                corr[TARGET]
                .drop(TARGET)
                .sort_values(ascending=False)
                .reset_index()
            )

            target_corr.columns = ["변수", "상관계수"]

            st.dataframe(
                target_corr,
                use_container_width=True,
                hide_index=True
            )

            fig = px.bar(
                target_corr,
                x="변수",
                y="상관계수",
                title="Correlation with Tool Wear"
            )

            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("제품 유형별 마모 비교")

        if "Type" in df.columns and TARGET in df.columns:
            fig = px.box(
                df,
                x="Type",
                y=TARGET,
                color="Type",
                title="Tool Wear by Product Type"
            )

            st.plotly_chart(fig, use_container_width=True)

            type_stats = (
                df.groupby("Type", observed=True)[TARGET]
                .agg(["count", "mean", "median", "std", "max"])
                .reset_index()
            )

            type_stats.columns = [
                "제품 유형", "데이터 수", "평균", "중앙값", "표준편차", "최댓값"
            ]

            st.dataframe(
                type_stats,
                use_container_width=True,
                hide_index=True
            )

        st.divider()
        st.subheader("설비 작동 변수와 공구 마모의 관계")

        scatter_features = [
            c for c in NUMERIC_FEATURES
            if c in df.columns and c != TARGET
        ]

        selected_scatter = st.selectbox(
            "분석할 변수",
            scatter_features
        )

        fig = px.scatter(
            df,
            x=selected_scatter,
            y=TARGET,
            color="Type" if "Type" in df.columns else None,
            trendline=None,
            title=f"{selected_scatter} vs Tool Wear"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("마모 시간 분포 및 기초통계")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("평균", f"{df[TARGET].mean():.2f}")
        c2.metric("중앙값", f"{df[TARGET].median():.2f}")
        c3.metric("표준편차", f"{df[TARGET].std():.2f}")
        c4.metric("최댓값", f"{df[TARGET].max():.2f}")

        fig = px.histogram(
            df,
            x=TARGET,
            nbins=40,
            marginal="box",
            title="Tool Wear Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 10. AI 모델 실험실
# =========================================================

elif menu == "🧠 AI 모델 실험실":

    st.title("AI 모델 실험실")
    st.caption("선형회귀 모델 학습, 변수 조합 실험 및 성능 평가")

    df = get_dataframe()

    if df is None:
        st.warning("먼저 데이터를 불러오세요.")
    elif TARGET not in df.columns:
        st.error(f"데이터에 목표 변수 {TARGET}가 없습니다.")
    else:

        available_features = [
            c for c in ALL_FEATURES
            if c in df.columns
        ]

        st.subheader("학습 조건 설정")

        selected_features = st.multiselect(
            "독립변수 선택",
            available_features,
            default=available_features
        )

        test_size = st.slider(
            "테스트 데이터 비율",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05
        )

        experiment_name = st.text_input(
            "실험 이름",
            value=f"실험 {datetime.now().strftime('%m%d-%H%M')}"
        )

        if st.button("모델 학습 및 평가", use_container_width=True):

            if not selected_features:
                st.error("독립변수를 하나 이상 선택하세요.")
            else:
                try:
                    model, metrics, test_result = train_model(
                        df,
                        selected_features,
                        test_size
                    )

                    st.session_state.model = model
                    st.session_state.features = selected_features
                    st.session_state.evaluation = metrics
                    st.session_state.test_data = test_result
                    st.session_state.current_experiment = experiment_name

                    save_experiment(
                        experiment_name,
                        selected_features,
                        test_size,
                        metrics["MAE"],
                        metrics["MSE"],
                        metrics["RMSE"],
                        metrics["R2"]
                    )

                    st.success("모델 학습과 평가가 완료되었습니다.")

                except Exception as e:
                    st.error(f"모델 학습 오류: {e}")

        if st.session_state.evaluation is not None:

            metrics = st.session_state.evaluation
            test_result = st.session_state.test_data

            st.divider()
            st.subheader("모델 성능 평가")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("MAE", f"{metrics['MAE']:.3f}")
            c2.metric("MSE", f"{metrics['MSE']:.3f}")
            c3.metric("RMSE", f"{metrics['RMSE']:.3f}")
            c4.metric("R²", f"{metrics['R2']:.4f}")

            st.caption(
                "MAE·RMSE는 예측 오차의 크기를, R²는 모델이 데이터의 변동을 설명하는 정도를 나타냅니다."
            )

            st.divider()
            st.subheader("실제값과 예측값 비교")

            fig = px.scatter(
                test_result,
                x="실제 마모 시간",
                y="예측 마모 시간",
                title="Actual vs Predicted"
            )

            min_val = min(
                test_result["실제 마모 시간"].min(),
                test_result["예측 마모 시간"].min()
            )

            max_val = max(
                test_result["실제 마모 시간"].max(),
                test_result["예측 마모 시간"].max()
            )

            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="이상적인 예측선"
            ))

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("잔차 분석")

            fig = px.scatter(
                test_result,
                x="예측 마모 시간",
                y="잔차",
                title="Residual Plot"
            )

            fig.add_hline(y=0, line_dash="dash")
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("회귀계수 분석")

            try:
                preprocessor = st.session_state.model.named_steps["preprocessor"]
                regressor = st.session_state.model.named_steps["regressor"]

                feature_names = preprocessor.get_feature_names_out()

                coef_df = pd.DataFrame({
                    "변수": feature_names,
                    "회귀계수": regressor.coef_
                })

                coef_df["절댓값"] = coef_df["회귀계수"].abs()
                coef_df = coef_df.sort_values(
                    "절댓값",
                    ascending=False
                )

                st.dataframe(
                    coef_df.drop(columns="절댓값"),
                    use_container_width=True,
                    hide_index=True
                )

                fig = px.bar(
                    coef_df,
                    x="변수",
                    y="회귀계수",
                    title="Regression Coefficients"
                )

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f"회귀계수 표시 중 문제가 발생했습니다: {e}")

        st.divider()
        st.subheader("실험 결과 비교")

        experiments = load_experiments()

        if not experiments.empty:
            st.dataframe(
                experiments,
                use_container_width=True,
                hide_index=True
            )

            fig = px.bar(
                experiments.sort_values("id"),
                x="experiment_name",
                y="r2",
                title="Experiment R² Comparison",
                hover_data=["mae", "rmse", "test_size"]
            )

            st.plotly_chart(fig, use_container_width=True)

            best = experiments.loc[experiments["r2"].idxmax()]

            st.info(
                f"현재 저장된 실험 중 가장 높은 R²: "
                f"{best['experiment_name']} ({best['r2']:.4f})"
            )

            delete_exp_id = st.number_input(
                "삭제할 실험 ID",
                min_value=1,
                step=1,
                key="delete_exp_id"
            )

            if st.button("선택 실험 기록 삭제"):
                delete_record("experiments", delete_exp_id)
                st.success("실험 기록을 삭제했습니다.")
                st.rerun()
        else:
            st.info("저장된 실험 결과가 없습니다.")


# =========================================================
# 11. 공구 마모 예측
# =========================================================

elif menu == "⚙️ 공구 마모 예측":

    st.title("공구 마모 예측")
    st.caption("설비 작동 조건을 입력해 공구 마모 시간을 예측합니다.")

    df = get_dataframe()

    if df is None:
        st.warning("먼저 데이터를 불러오세요.")
    elif st.session_state.model is None:
        st.warning("AI 모델 실험실에서 먼저 모델을 학습하세요.")
    else:

        model = st.session_state.model
        features = st.session_state.features

        st.subheader("설비 조건 입력")

        input_values = {}

        col1, col2 = st.columns(2)

        for i, feature in enumerate(features):
            container = col1 if i % 2 == 0 else col2

            with container:

                if feature == "Type":
                    input_values[feature] = st.selectbox(
                        "제품 유형",
                        ["L", "M", "H"]
                    )

                else:
                    series = pd.to_numeric(
                        df[feature],
                        errors="coerce"
                    ).dropna()

                    if series.empty:
                        input_values[feature] = 0.0
                    else:
                        minimum = float(series.min())
                        maximum = float(series.max())
                        median = float(series.median())

                        input_values[feature] = st.number_input(
                            feature,
                            min_value=minimum,
                            max_value=maximum,
                            value=median
                        )

        if st.button("공구 마모 예측 실행", use_container_width=True):

            input_df = pd.DataFrame([input_values])

            try:
                prediction = float(model.predict(input_df)[0])

                save_prediction(
                    prediction,
                    input_values,
                    "직접 예측"
                )

                st.success("예측이 완료되었습니다.")

                st.metric(
                    "예측 공구 마모 시간",
                    f"{prediction:.2f} min"
                )

                st.caption(
                    "이 값은 모델이 추정한 공구 마모 시간입니다. "
                    "실제 교체 시점이나 잔여 수명을 직접 나타내지는 않습니다."
                )

            except Exception as e:
                st.error(f"예측 오류: {e}")

        st.divider()
        st.subheader("CSV 일괄 예측")

        batch_file = st.file_uploader(
            "예측할 설비 데이터 CSV 업로드",
            type=["csv"],
            key="batch_upload"
        )

        if batch_file is not None:

            try:
                batch_df = pd.read_csv(batch_file)

                missing_features = [
                    f for f in features
                    if f not in batch_df.columns
                ]

                if missing_features:
                    st.error(
                        f"CSV에 필요한 변수가 없습니다: {missing_features}"
                    )
                else:
                    st.dataframe(batch_df.head(), use_container_width=True)

                    if st.button("일괄 예측 실행"):

                        batch_input = batch_df[features].copy()
                        batch_predictions = model.predict(batch_input)

                        result_df = batch_df.copy()
                        result_df["예측 공구 마모 시간 [min]"] = batch_predictions

                        for _, row in result_df.iterrows():
                            input_dict = {
                                f: row[f] for f in features
                            }

                            save_prediction(
                                row["예측 공구 마모 시간 [min]"],
                                input_dict,
                                "CSV 일괄 예측"
                            )

                        st.success(
                            f"{len(result_df)}개 데이터의 예측을 완료했습니다."
                        )

                        st.dataframe(
                            result_df,
                            use_container_width=True
                        )

                        st.download_button(
                            "예측 결과 CSV 다운로드",
                            data=result_df.to_csv(
                                index=False
                            ).encode("utf-8-sig"),
                            file_name="tool_wear_predictions.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"일괄 예측 오류: {e}")

        st.divider()
        st.subheader("예측 이력")

        predictions = load_predictions()

        if not predictions.empty:
            st.dataframe(
                predictions,
                use_container_width=True,
                hide_index=True
            )

            delete_id = st.number_input(
                "삭제할 예측 기록 ID",
                min_value=1,
                step=1,
                key="delete_prediction_id"
            )

            if st.button("예측 기록 삭제"):
                delete_record("predictions", delete_id)
                st.success("예측 기록을 삭제했습니다.")
                st.rerun()
        else:
            st.info("예측 이력이 없습니다.")


# =========================================================
# 12. 설비 및 정비 관리
# =========================================================

elif menu == "🔧 설비 및 정비 관리":

    st.title("설비 및 정비 관리")
    st.caption("설비 정비 이력 등록, 조회, 수정 및 삭제")

    st.subheader("정비 기록 등록")

    with st.form("maintenance_form"):

        equipment = st.text_input(
            "설비명 또는 설비 ID",
            value="Machine-001"
        )

        maintenance_type = st.selectbox(
            "정비 유형",
            [
                "공구 교체",
                "정기 점검",
                "부품 교체",
                "윤활 및 청소",
                "기타"
            ]
        )

        wear_value = st.number_input(
            "정비 당시 공구 마모 시간 [min]",
            min_value=0.0,
            value=0.0
        )

        description = st.text_area(
            "정비 내용 및 특이사항"
        )

        submitted = st.form_submit_button("정비 기록 저장")

        if submitted:
            if not equipment.strip():
                st.error("설비명을 입력하세요.")
            else:
                save_maintenance(
                    equipment,
                    maintenance_type,
                    wear_value,
                    description
                )

                st.success("정비 기록을 저장했습니다.")
                st.rerun()

    st.divider()
    st.subheader("정비 이력 조회")

    maintenance = load_maintenance()

    if not maintenance.empty:

        st.dataframe(
            maintenance,
            use_container_width=True,
            hide_index=True
        )

        st.divider()
        st.subheader("정비 기록 수정")

        record_options = {
            f"ID {row['id']} | {row['equipment']} | {row['created_at']}": int(row["id"])
            for _, row in maintenance.iterrows()
        }

        selected_label = st.selectbox(
            "수정할 기록 선택",
            list(record_options.keys())
        )

        selected_id = record_options[selected_label]
        selected_row = maintenance[
            maintenance["id"] == selected_id
        ].iloc[0]

        with st.form("edit_maintenance_form"):

            edit_equipment = st.text_input(
                "설비명",
                value=str(selected_row["equipment"])
            )

            edit_type = st.selectbox(
                "정비 유형",
                [
                    "공구 교체",
                    "정기 점검",
                    "부품 교체",
                    "윤활 및 청소",
                    "기타"
                ],
                index=(
                    [
                        "공구 교체",
                        "정기 점검",
                        "부품 교체",
                        "윤활 및 청소",
                        "기타"
                    ].index(selected_row["maintenance_type"])
                    if selected_row["maintenance_type"] in [
                        "공구 교체",
                        "정기 점검",
                        "부품 교체",
                        "윤활 및 청소",
                        "기타"
                    ] else 0
                )
            )

            edit_wear = st.number_input(
                "마모 시간",
                min_value=0.0,
                value=float(selected_row["wear_value"])
            )

            edit_description = st.text_area(
                "정비 내용",
                value=str(selected_row["description"])
            )

            edit_submitted = st.form_submit_button("수정 내용 저장")

            if edit_submitted:
                update_maintenance(
                    selected_id,
                    edit_equipment,
                    edit_type,
                    edit_wear,
                    edit_description
                )

                st.success("정비 기록을 수정했습니다.")
                st.rerun()

        st.divider()
        st.subheader("정비 기록 삭제")

        delete_maintenance_id = st.selectbox(
            "삭제할 정비 기록 ID",
            maintenance["id"].astype(int).tolist()
        )

        if st.button("선택한 정비 기록 삭제"):
            delete_record("maintenance", delete_maintenance_id)
            st.success("정비 기록을 삭제했습니다.")
            st.rerun()

        st.download_button(
            "정비 이력 CSV 다운로드",
            data=maintenance.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="maintenance_history.csv",
            mime="text/csv"
        )

    else:
        st.info("등록된 정비 기록이 없습니다.")

    st.divider()
    st.subheader("예측 마모와 실제 정비 기록 비교")

    predictions = load_predictions()
    maintenance = load_maintenance()

    if not predictions.empty and not maintenance.empty:

        pred_summary = (
            predictions.groupby("source")["prediction"]
            .agg(["count", "mean", "max"])
            .reset_index()
        )

        st.write("예측 기록 요약")
        st.dataframe(pred_summary, use_container_width=True)

        st.write("실제 정비 기록의 마모 시간")
        st.dataframe(
            maintenance[[
                "created_at",
                "equipment",
                "wear_value",
                "maintenance_type"
            ]],
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "예측 기록과 정비 기록은 서로 다른 시점과 설비에서 발생했을 수 있으므로, "
            "단순 평균 비교만으로 모델의 정비 효과를 판단할 수는 없습니다."
        )

    else:
        st.info("예측 기록과 정비 기록이 모두 있어야 비교할 수 있습니다.")


# =========================================================
# 13. 분석 보고서
# =========================================================

elif menu == "📄 분석 보고서":

    st.title("분석 보고서")
    st.caption("데이터 분석, 모델 평가, 실험 및 정비 기록 요약")

    df = get_dataframe()
    predictions = load_predictions()
    experiments = load_experiments()
    maintenance = load_maintenance()

    if df is None:
        st.warning("데이터가 등록되지 않았습니다.")
    else:

        st.subheader("1. 데이터 요약")

        c1, c2, c3 = st.columns(3)
        c1.metric("데이터 행 수", f"{len(df):,}")
        c2.metric("변수 수", f"{len(df.columns)}")
        c3.metric("결측치 수", f"{int(df.isnull().sum().sum()):,}")

        st.subheader("2. 공구 마모 기초통계")

        if TARGET in df.columns:
            st.dataframe(
                df[TARGET].describe().to_frame("공구 마모 시간"),
                use_container_width=True
            )

        st.subheader("3. AI 모델 평가")

        if st.session_state.evaluation is not None:

            metrics = st.session_state.evaluation

            st.dataframe(
                pd.DataFrame([metrics]),
                use_container_width=True
            )

            st.write("사용 변수:", ", ".join(st.session_state.features))

        else:
            st.info("아직 학습된 모델 평가 결과가 없습니다.")

        st.subheader("4. 실험 기록")
        st.dataframe(
            experiments,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("5. 예측 기록")
        st.dataframe(
            predictions,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("6. 정비 기록")
        st.dataframe(
            maintenance,
            use_container_width=True,
            hide_index=True
        )

        report_lines = [
            "산업 설비 공구 마모 분석 보고서",
            "=" * 45,
            f"보고서 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "[데이터 요약]",
            f"행 개수: {len(df)}",
            f"열 개수: {len(df.columns)}",
            f"결측치 수: {int(df.isnull().sum().sum())}",
            f"중복 행 수: {int(df.duplicated().sum())}",
            "",
            "[공구 마모 기초통계]"
        ]

        if TARGET in df.columns:
            report_lines.extend([
                f"평균: {df[TARGET].mean():.3f}",
                f"중앙값: {df[TARGET].median():.3f}",
                f"표준편차: {df[TARGET].std():.3f}",
                f"최댓값: {df[TARGET].max():.3f}"
            ])

        report_lines.extend([
            "",
            "[모델 평가]"
        ])

        if st.session_state.evaluation is not None:
            for key, value in st.session_state.evaluation.items():
                report_lines.append(f"{key}: {value:.5f}")
            report_lines.append(
                "사용 변수: " + ", ".join(st.session_state.features)
            )
        else:
            report_lines.append("모델 평가 기록 없음")

        report_lines.extend([
            "",
            f"저장된 실험 수: {len(experiments)}",
            f"예측 기록 수: {len(predictions)}",
            f"정비 기록 수: {len(maintenance)}"
        ])

        report_text = "\n".join(report_lines)

        st.download_button(
            "분석 보고서 TXT 다운로드",
            data=report_text.encode("utf-8-sig"),
            file_name="industrial_ai_report.txt",
            mime="text/plain"
        )

        st.download_button(
            "현재 데이터 CSV 다운로드",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="analysis_data.csv",
            mime="text/csv"
        )

        if not experiments.empty:
            st.download_button(
                "실험 기록 CSV 다운로드",
                data=experiments.to_csv(index=False).encode("utf-8-sig"),
                file_name="experiment_history.csv",
                mime="text/csv"
            )

st.sidebar.divider()
st.sidebar.caption("Industrial AI Tool Wear Monitoring")
st.sidebar.caption("Prototype for educational use")


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from datetime import datetime
import os

# =====================================================
# 1. 기본 설정 및 디자인
# =====================================================

st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background-color: #111c2e;
}
[data-testid="stMetric"] {
    background-color: #17243a;
    border: 1px solid #293b55;
    padding: 15px;
    border-radius: 10px;
}
div[data-testid="stMetricValue"] {
    color: #60a5fa;
}
.stButton > button {
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 7px;
}
.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}
h1, h2, h3 {
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 2. 데이터 및 세션 초기화
# =====================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "model" not in st.session_state:
    st.session_state.model = None

if "features" not in st.session_state:
    st.session_state.features = []

if "experiments" not in st.session_state:
    st.session_state.experiments = []

if "maintenance_log" not in st.session_state:
    st.session_state.maintenance_log = []

if "evaluation" not in st.session_state:
    st.session_state.evaluation = None

if "test_data" not in st.session_state:
    st.session_state.test_data = None

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# =====================================================
# 3. 공통 변수
# =====================================================

TARGET = "Tool wear [min]"

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]"
]

ALL_FEATURES = ["Type"] + NUMERIC_FEATURES


def get_data():
    return st.session_state.df


def get_model():
    return st.session_state.model


def make_pipeline(features, data):
    categorical = [
        col for col in features
        if data[col].dtype == "object"
        or isinstance(data[col].dtype, pd.CategoricalDtype)
    ]

    numeric = [
        col for col in features
        if col not in categorical
    ]

    transformers = []

    if numeric:
        transformers.append(
            ("numeric", "passthrough", numeric)
        )

    if categorical:
        transformers.append(
            ("categorical",
             OneHotEncoder(handle_unknown="ignore"),
             categorical)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])


def clean_data(data):
    data = data.copy()

    for col in data.columns:
        if col in ALL_FEATURES + [TARGET]:
            if col != "Type":
                data[col] = pd.to_numeric(
                    data[col], errors="coerce"
                )

    return data


# =====================================================
# 4. 사이드바 메뉴
# =====================================================

st.sidebar.markdown("# ⚙️ PM SYSTEM")
st.sidebar.caption("INDUSTRIAL ANALYTICS PLATFORM")
st.sidebar.divider()

menu = st.sidebar.radio(
    "MAIN MENU",
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

st.sidebar.markdown("### SYSTEM STATUS")

if get_data() is not None:
    st.sidebar.success("DATA CONNECTED")
else:
    st.sidebar.warning("DATA NOT CONNECTED")

if get_model() is not None:
    st.sidebar.success("MODEL TRAINED")
else:
    st.sidebar.info("MODEL NOT TRAINED")

st.sidebar.caption("Predictive Maintenance v1.0")


# =====================================================
# 5. 공통 데이터 안내
# =====================================================

df = get_data()

if df is not None:
    st.session_state.df = clean_data(df)
    df = get_data()


# =====================================================
# 6. 메인 대시보드
# =====================================================

if menu == "🏠 메인 대시보드":

    st.title("INDUSTRIAL MONITORING CENTER")
    st.caption("AI 기반 공구 마모 예측 및 설비 관리 통합 시스템")

    st.divider()

    if df is None:
        st.warning("분석할 데이터가 없습니다. 데이터 관리 메뉴에서 CSV 파일을 업로드하세요.")
    else:
        total_rows = len(df)
        total_columns = len(df.columns)
        missing = int(df.isnull().sum().sum())
        model_status = "학습 완료" if get_model() else "대기 중"

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("전체 데이터", f"{total_rows:,} rows")
        c2.metric("데이터 변수", f"{total_columns} columns")
        c3.metric("결측 데이터", f"{missing:,}")
        c4.metric("AI 모델 상태", model_status)

        st.divider()

        left, right = st.columns(2)

        with left:
            st.subheader("제품 유형별 데이터")
            if "Type" in df.columns:
                counts = df["Type"].value_counts().reset_index()
                counts.columns = ["Type", "Count"]

                fig = px.bar(
                    counts,
                    x="Type",
                    y="Count",
                    color="Type",
                    title="Product Type Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("공구 마모 시간 분포")
            if TARGET in df.columns:
                fig = px.histogram(
                    df,
                    x=TARGET,
                    nbins=30,
                    title="Tool Wear Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("최근 예측 기록")

        if st.session_state.prediction_history:
            st.dataframe(
                pd.DataFrame(
                    st.session_state.prediction_history
                ).tail(5),
                use_container_width=True
            )
        else:
            st.info("아직 예측 기록이 없습니다.")


# =====================================================
# 7. 데이터 관리
# =====================================================

elif menu == "📁 데이터 관리":

    st.title("DATA MANAGEMENT")
    st.caption("CSV 업로드 및 데이터 전처리")

    uploaded = st.file_uploader(
        "CSV 데이터 업로드",
        type=["csv"]
    )

    if uploaded is not None:
        try:
            loaded = pd.read_csv(uploaded)
            st.session_state.df = clean_data(loaded)
            st.session_state.model = None
            st.session_state.evaluation = None
            st.success("데이터 업로드 완료")
        except Exception as e:
            st.error(f"파일을 읽을 수 없습니다: {e}")

    if st.button("기본 AI4I 데이터 불러오기"):
        if os.path.exists("ai4i2020.csv"):
            st.session_state.df = clean_data(
                pd.read_csv("ai4i2020.csv")
            )
            st.session_state.model = None
            st.session_state.evaluation = None
            st.success("기본 데이터 로드 완료")
        else:
            st.error("저장소에 ai4i2020.csv 파일이 없습니다.")

    df = get_data()

    if df is not None:

        st.divider()

        st.subheader("1. 데이터 미리보기")
        st.dataframe(df.head(100), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("행 개수", len(df))
        c2.metric("열 개수", len(df.columns))
        c3.metric("중복 행", int(df.duplicated().sum()))

        st.subheader("2. 데이터 유형")
        types = pd.DataFrame({
            "Column": df.columns,
            "Data Type": [str(x) for x in df.dtypes]
        })
        st.dataframe(types, use_container_width=True)

        st.subheader("3. 결측치 분석")

        missing = df.isnull().sum()

        missing_df = pd.DataFrame({
            "Column": missing.index,
            "Missing Count": missing.values,
            "Missing Rate (%)":
                (missing.values / len(df) * 100).round(2)
        })

        st.dataframe(missing_df, use_container_width=True)

        if missing.sum() > 0:
            if st.button("결측치가 있는 행 제거"):
                st.session_state.df = df.dropna().copy()
                st.success("결측치 행 제거 완료")
                st.rerun()
        else:
            st.success("결측치가 없습니다.")

        st.subheader("4. 이상치 탐지 (IQR)")

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_cols:
            out_col = st.selectbox(
                "검사할 변수",
                numeric_cols
            )

            q1 = df[out_col].quantile(0.25)
            q3 = df[out_col].quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outliers = df[
                (df[out_col] < lower)
                | (df[out_col] > upper)
            ]

            c1, c2, c3 = st.columns(3)
            c1.metric("하한", f"{lower:.2f}")
            c2.metric("상한", f"{upper:.2f}")
            c3.metric("이상치 개수", len(outliers))

            st.dataframe(
                outliers.head(100),
                use_container_width=True
            )

            st.caption(
                "IQR 이상치는 통계적 기준을 벗어난 관측값이며, "
                "반드시 오류 데이터라는 뜻은 아닙니다."
            )

        st.subheader("5. 데이터 필터링")

        filtered = df.copy()

        if "Type" in df.columns:
            types = st.multiselect(
                "제품 유형 선택",
                df["Type"].dropna().unique(),
                default=list(df["Type"].dropna().unique())
            )
            filtered = filtered[filtered["Type"].isin(types)]

        for col in NUMERIC_FEATURES:
            if col in filtered.columns:
                minimum = float(filtered[col].min())
                maximum = float(filtered[col].max())

                if minimum < maximum:
                    selected = st.slider(
                        col,
                        min_value=minimum,
                        max_value=maximum,
                        value=(minimum, maximum)
                    )

                    filtered = filtered[
                        filtered[col].between(
                            selected[0], selected[1]
                        )
                    ]

        st.write(f"필터링 결과: {len(filtered):,}개 행")
        st.dataframe(filtered, use_container_width=True)

        st.download_button(
            "필터링 데이터 CSV 다운로드",
            filtered.to_csv(index=False).encode("utf-8-sig"),
            "filtered_data.csv",
            "text/csv"
        )


# =====================================================
# 8. 데이터 분석 및 시각화
# =====================================================

elif menu == "📊 데이터 분석":

    st.title("DATA ANALYTICS")
    st.caption("데이터 통계 및 시각화 분석")

    if df is None:
        st.warning("먼저 데이터를 업로드하세요.")
    else:

        st.subheader("6. 상관관계 히트맵")

        numeric = df.select_dtypes(
            include=np.number
        )

        corr = numeric.corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix"
        )

        st.plotly_chart(fig, use_container_width=True)

        if TARGET in corr.columns:
            st.subheader("공구 마모 시간과의 상관관계")

            wear_corr = corr[TARGET].drop(TARGET).sort_values(
                ascending=False
            )

            st.dataframe(
                wear_corr.rename("Correlation"),
                use_container_width=True
            )

        st.divider()

        st.subheader("7. 제품 유형별 마모 시간 비교")

        if "Type" in df.columns and TARGET in df.columns:

            fig = px.box(
                df,
                x="Type",
                y=TARGET,
                color="Type",
                points="outliers",
                title="Tool Wear by Product Type"
            )

            st.plotly_chart(fig, use_container_width=True)

            type_stats = df.groupby("Type")[TARGET].agg(
                ["count", "mean", "median", "std"]
            )

            st.dataframe(type_stats, use_container_width=True)

        st.divider()

        st.subheader("8. 변수별 산점도")

        available = [
            c for c in NUMERIC_FEATURES
            if c in df.columns
        ]

        if available and TARGET in df.columns:

            selected_x = st.selectbox(
                "분석할 독립변수",
                available
            )

            fig = px.scatter(
                df,
                x=selected_x,
                y=TARGET,
                color="Type" if "Type" in df.columns else None,
                trendline="ols",
                title=f"{selected_x} vs Tool Wear"
            )

            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("9. 데이터 분포 및 기초통계")

        selected_stat = st.selectbox(
            "통계 분석 변수",
            numeric.columns
        )

        values = df[selected_stat].dropna()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("평균", f"{values.mean():.2f}")
        c2.metric("중앙값", f"{values.median():.2f}")
        c3.metric("표준편차", f"{values.std():.2f}")
        c4.metric("최댓값", f"{values.max():.2f}")

        fig = px.histogram(
            df,
            x=selected_stat,
            nbins=40,
            marginal="box",
            title=f"{selected_stat} Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)


# =====================================================
# 9. AI 모델 실험실
# =====================================================

elif menu == "🧠 AI 모델 실험실":

    st.title("AI MODEL LAB")
    st.caption("Linear Regression Training & Evaluation")

    if df is None:
        st.warning("먼저 데이터를 업로드하세요.")

    elif TARGET not in df.columns:
        st.error(f"종속변수 {TARGET} 컬럼이 필요합니다.")

    else:

        available = [
            c for c in ALL_FEATURES
            if c in df.columns
        ]

        st.subheader("10. 독립변수 선택")

        selected_features = st.multiselect(
            "모델 학습에 사용할 변수",
            available,
            default=available
        )

        st.subheader("11. 데이터 분할 설정")

        test_ratio = st.slider(
            "테스트 데이터 비율 (%)",
            10, 40, 20, 5
        )

        st.caption(
            f"학습 데이터 {100-test_ratio}% / "
            f"테스트 데이터 {test_ratio}%"
        )

        st.subheader("12. 선형회귀 모델 학습")

        if st.button("🚀 모델 학습 시작", use_container_width=True):

            if not selected_features:
                st.error("독립변수를 한 개 이상 선택하세요.")

            else:
                model_data = df[
                    selected_features + [TARGET]
                ].dropna().copy()

                X = model_data[selected_features]
                y = model_data[TARGET]

                if len(model_data) < 10:
                    st.error("학습에 필요한 데이터가 부족합니다.")
                else:
                    try:
                        X_train, X_test, y_train, y_test = (
                            train_test_split(
                                X,
                                y,
                                test_size=test_ratio / 100,
                                random_state=42
                            )
                        )

                        model = make_pipeline(
                            selected_features,
                            model_data
                        )

                        model.fit(X_train, y_train)

                        pred = model.predict(X_test)

                        mae = mean_absolute_error(
                            y_test, pred
                        )
                        mse = mean_squared_error(
                            y_test, pred
                        )
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, pred)

                        st.session_state.model = model
                        st.session_state.features = (
                            selected_features
                        )

                        st.session_state.evaluation = {
                            "MAE": mae,
                            "MSE": mse,
                            "RMSE": rmse,
                            "R2": r2,
                            "Test Ratio": test_ratio
                        }

                        st.session_state.test_data = pd.DataFrame({
                            "Actual": y_test.to_numpy(),
                            "Predicted": pred
                        })

                        experiment = {
                            "Date": datetime.now().strftime(
                                "%Y-%m-%d %H:%M"
                            ),
                            "Features": ", ".join(
                                selected_features
                            ),
                            "Test Ratio": test_ratio,
                            "MAE": mae,
                            "RMSE": rmse,
                            "R2": r2
                        }

                        st.session_state.experiments.append(
                            experiment
                        )

                        st.success("모델 학습 및 평가 완료!")

                    except Exception as e:
                        st.error(f"모델 학습 오류: {e}")

        # 평가 결과
        evaluation = st.session_state.evaluation

        if evaluation is not None:

            st.divider()
            st.subheader("13. 모델 평가 결과")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "MAE",
                f"{evaluation['MAE']:.3f}"
            )
            c2.metric(
                "MSE",
                f"{evaluation['MSE']:.3f}"
            )
            c3.metric(
                "RMSE",
                f"{evaluation['RMSE']:.3f}"
            )
            c4.metric(
                "R²",
                f"{evaluation['R2']:.3f}"
            )

            st.caption(
                "MAE·RMSE는 예측 오차의 크기를 나타내며 "
                "낮을수록 좋습니다. R²는 설명력을 나타내며 "
                "일반적으로 1에 가까울수록 좋습니다."
            )

            st.subheader("14. 실제값과 예측값 비교")

            test = st.session_state.test_data

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=test["Actual"],
                y=test["Predicted"],
                mode="markers",
                name="Prediction"
            ))

            low = min(
                test["Actual"].min(),
                test["Predicted"].min()
            )
            high = max(
                test["Actual"].max(),
                test["Predicted"].max()
            )

            fig.add_trace(go.Scatter(
                x=[low, high],
                y=[low, high],
                mode="lines",
                name="Ideal Prediction"
            ))

            fig.update_layout(
                xaxis_title="Actual Tool Wear",
                yaxis_title="Predicted Tool Wear"
            )

            st.plotly_chart(fig, use_container_width=True)

            residual = (
                test["Actual"] - test["Predicted"]
            )

            residual_df = pd.DataFrame({
                "Predicted": test["Predicted"],
                "Residual": residual
            })

            fig = px.scatter(
                residual_df,
                x="Predicted",
                y="Residual",
                title="Residual Analysis"
            )

            fig.add_hline(y=0, line_dash="dash")

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("모델 회귀계수")

            fitted = get_model()
            feature_names = (
                fitted.named_steps["preprocessor"]
                .get_feature_names_out()
            )

            coefficients = (
                fitted.named_steps["regressor"].coef_
            )

            coef_df = pd.DataFrame({
                "Variable": feature_names,
                "Coefficient": coefficients
            })

            st.dataframe(
                coef_df,
                use_container_width=True
            )

        st.divider()

        st.subheader("모델 실험 기록")

        if st.session_state.experiments:
            exp_df = pd.DataFrame(
                st.session_state.experiments
            )

            st.dataframe(exp_df, use_container_width=True)

            st.download_button(
                "실험 결과 CSV 다운로드",
                exp_df.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                "model_experiments.csv",
                "text/csv"
            )

            st.subheader("실험별 R² 비교")

            fig = px.bar(
                exp_df.reset_index(),
                x="index",
                y="R2",
                title="Experiment R² Comparison"
            )

            st.plotly_chart(fig, use_container_width=True)


# =====================================================
# 10. 공구 마모 예측 센터
# =====================================================

elif menu == "⚙️ 공구 마모 예측":

    st.title("TOOL WEAR PREDICTION CENTER")
    st.caption("Trained Linear Regression Model")

    model = get_model()

    if model is None:
        st.warning("AI 모델 실험실에서 먼저 모델을 학습하세요.")

    else:

        df = get_data()
        features = st.session_state.features

        st.subheader("설비 작동 조건 입력")

        with st.form("prediction_form"):

            input_values = {}

            cols = st.columns(2)

            for i, feature in enumerate(features):

                with cols[i % 2]:

                    if feature == "Type":
                        input_values[feature] = st.selectbox(
                            "제품 유형",
                            ["L", "M", "H"]
                        )

                    else:
                        default = float(df[feature].median())

                        minimum = float(df[feature].min())
                        maximum = float(df[feature].max())

                        input_values[feature] = st.number_input(
                            feature,
                            min_value=minimum,
                            max_value=maximum,
                            value=default
                        )

            submitted = st.form_submit_button(
                "공구 마모 시간 예측",
                use_container_width=True
            )

        if submitted:

            input_df = pd.DataFrame(
                [input_values],
                columns=features
            )

            prediction = float(
                model.predict(input_df)[0]
            )

            st.success("예측 완료")

            st.metric(
                "예측 공구 마모 시간",
                f"{prediction:.2f} min"
            )

            record = {
                **input_values,
                "Predicted Tool Wear": prediction,
                "Date": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            }

            st.session_state.prediction_history.append(
                record
            )

            st.info(
                "이 값은 학습 데이터의 관계를 바탕으로 계산한 "
                "공구 마모 시간 추정치입니다. 실제 마모 상태나 "
                "잔여 수명을 직접 측정한 값은 아닙니다."
            )

        st.divider()

        st.subheader("여러 조건 일괄 예측")

        batch_file = st.file_uploader(
            "예측할 조건이 담긴 CSV 업로드",
            type=["csv"],
            key="batch_upload"
        )

        if batch_file is not None:

            try:
                batch = pd.read_csv(batch_file)

                missing_features = [
                    f for f in features
                    if f not in batch.columns
                ]

                if missing_features:
                    st.error(
                        f"필요한 컬럼이 없습니다: {missing_features}"
                    )
                else:
                    batch_pred = model.predict(
                        batch[features]
                    )

                    result = batch.copy()
                    result["Predicted Tool Wear"] = batch_pred

                    st.dataframe(
                        result,
                        use_container_width=True
                    )

                    st.download_button(
                        "일괄 예측 결과 CSV 다운로드",
                        result.to_csv(
                            index=False
                        ).encode("utf-8-sig"),
                        "batch_predictions.csv",
                        "text/csv"
                    )

            except Exception as e:
                st.error(f"예측 오류: {e}")

        st.divider()

        st.subheader("예측 이력")

        history = st.session_state.prediction_history

        if history:
            history_df = pd.DataFrame(history)

            st.dataframe(
                history_df,
                use_container_width=True
            )

            st.download_button(
                "예측 기록 CSV 다운로드",
                history_df.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                "prediction_history.csv",
                "text/csv"
            )


# =====================================================
# 11. 설비 및 정비 관리
# =====================================================

elif menu == "🔧 설비 및 정비 관리":

    st.title("MAINTENANCE MANAGEMENT")
    st.caption("설비 정비 이력 및 예측 기록 관리")

    st.subheader("정비 이력 등록")

    with st.form("maintenance_form"):

        machine = st.text_input(
            "설비 ID",
            placeholder="예: CNC-001"
        )

        tool_name = st.text_input(
            "공구 이름",
            placeholder="예: Cutting Tool A"
        )

        wear_value = st.number_input(
            "정비 당시 확인한 마모 시간 (min)",
            min_value=0.0,
            value=0.0
        )

        maintenance_type = st.selectbox(
            "정비 유형",
            ["공구 교체", "공구 점검", "설비 점검", "기타"]
        )

        note = st.text_area("정비 내용")

        save = st.form_submit_button(
            "정비 기록 등록",
            use_container_width=True
        )

    if save:

        if machine.strip():

            st.session_state.maintenance_log.append({
                "Date": datetime.now().strftime(
                    "%Y-%m-%d %H:%M"
                ),
                "Machine ID": machine,
                "Tool": tool_name,
                "Wear Time": wear_value,
                "Type": maintenance_type,
                "Note": note
            })

            st.success("정비 기록 등록 완료")

        else:
            st.error("설비 ID를 입력하세요.")

    st.divider()

    st.subheader("정비 이력 조회")

    logs = st.session_state.maintenance_log

    if logs:

        log_df = pd.DataFrame(logs)

        st.dataframe(
            log_df,
            use_container_width=True
        )

        delete_index = st.selectbox(
            "삭제할 기록 선택",
            range(len(log_df)),
            format_func=lambda x:
                f"{x+1}. {log_df.iloc[x]['Machine ID']} "
                f"- {log_df.iloc[x]['Date']}"
        )

        if st.button("선택 기록 삭제"):
            st.session_state.maintenance_log.pop(
                delete_index
            )
            st.success("삭제 완료")
            st.rerun()

        st.download_button(
            "정비 이력 CSV 다운로드",
            log_df.to_csv(
                index=False
            ).encode("utf-8-sig"),
            "maintenance_history.csv",
            "text/csv"
        )

        st.subheader("예측 결과와 정비 이력 비교")

        predictions = st.session_state.prediction_history

        if predictions:

            pred_df = pd.DataFrame(predictions)

            st.write("최근 예측 결과")
            st.dataframe(
                pred_df.tail(10),
                use_container_width=True
            )

            st.caption(
                "예측 마모 시간과 실제 정비 기록을 함께 살펴볼 수 있습니다. "
                "두 값은 측정 시점과 기준이 다를 수 있으므로 "
                "동일한 시점의 값처럼 단순 비교하면 안 됩니다."
            )

        else:
            st.info("아직 예측 기록이 없습니다.")

    else:
        st.info("등록된 정비 기록이 없습니다.")


# =====================================================
# 12. 분석 보고서
# =====================================================

elif menu == "📄 분석 보고서":

    st.title("ANALYSIS REPORT")
    st.caption("데이터 및 머신러닝 분석 결과 통합")

    df = get_data()
    evaluation = st.session_state.evaluation

    if df is None:
        st.warning("데이터를 먼저 업로드하세요.")

    else:

        st.subheader("데이터 요약")

        st.write(f"- 데이터 행 수: {len(df):,}")
        st.write(f"- 데이터 열 수: {len(df.columns)}")
        st.write(f"- 결측치 수: {df.isnull().sum().sum():,}")

        if TARGET in df.columns:
            st.write(
                f"- 평균 공구 마모 시간: "
                f"{df[TARGET].mean():.2f} min"
            )

        st.divider()

        st.subheader("모델 평가 결과")

        if evaluation:

            for metric, value in evaluation.items():
                st.write(f"- {metric}: {value}")

        else:
            st.info("아직 학습된 모델이 없습니다.")

        st.divider()

        st.subheader("실험 기록")

        if st.session_state.experiments:
            st.dataframe(
                pd.DataFrame(
                    st.session_state.experiments
                ),
                use_container_width=True
            )

        report = [
            "PREDICTIVE MAINTENANCE ANALYSIS REPORT",
            f"Generated: {datetime.now()}",
            "",
            f"Data rows: {len(df)}",
            f"Data columns: {len(df.columns)}",
            f"Missing values: {df.isnull().sum().sum()}"
        ]

        if evaluation:
            report.append("")
            report.append("MODEL EVALUATION")

            for key, value in evaluation.items():
                report.append(f"{key}: {value}")

        st.download_button(
            "분석 보고서 TXT 다운로드",
            "\n".join(report),
            "analysis_report.txt",
            "text/plain"
        )

        if st.button("현재 분석 데이터 CSV 다운로드"):
            st.download_button(
                "CSV 저장",
                df.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                "analysis_data.csv",
                "text/csv"
            )


# =====================================================
# 13. 하단 정보
# =====================================================

st.sidebar.divider()
st.sidebar.caption("AI4I 2020 Predictive Maintenance")
st.sidebar.caption("Industrial Engineering Project")

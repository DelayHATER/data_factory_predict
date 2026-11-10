import streamlit as st

st.title("⚙️ 설비 관리 통합 센터")

st.write("AI 기반 공구 마모 예측 및 설비 관리 시스템")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("분석 시스템", "ONLINE")

with col2:
    st.metric("예측 모델", "준비 중")

with col3:
    st.metric("데이터 상태", "확인 필요")

st.divider()

st.subheader("시스템 안내")

st.info(
    "왼쪽 메뉴에서 데이터 관리, "
    "데이터 분석, AI 모델 실험실, "
    "공구 마모 예측 기능을 선택하세요."
)

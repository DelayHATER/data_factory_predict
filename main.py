
import streamlit as st

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

pages = {
    "MAIN": [
        st.Page("pages/dashboard.py", title="메인 대시보드",
                icon="🏠", default=True)
    ],
    "DATA": [
        st.Page("pages/data_manager.py", title="데이터 관리",
                icon="📁"),
        st.Page("pages/data_analysis.py", title="데이터 분석",
                icon="📊")
    ],
    "AI SYSTEM": [
        st.Page("pages/model_lab.py", title="AI 모델 실험실",
                icon="🧠"),
        st.Page("pages/prediction.py", title="공구 마모 예측",
                icon="⚙️")
    ],
    "MANAGEMENT": [
        st.Page("pages/maintenance.py", title="설비 및 정비 관리",
                icon="🔧"),
        st.Page("pages/reports.py", title="분석 보고서",
                icon="📄")
    ]
}

pg = st.navigation(pages)
pg.run()

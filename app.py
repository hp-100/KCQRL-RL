"""Streamlit 应用入口。"""

from __future__ import annotations

import streamlit as st

from src.database import DatabaseError, init_database

st.set_page_config(page_title="高考数学错题诊断与个性化变式题生成系统", layout="wide")

try:
    init_database()
except DatabaseError as exc:
    st.error(str(exc))
    st.stop()

student_page = st.Page("pages/student_management.py", title="学生管理", icon="👩‍🎓")
wrong_question_page = st.Page("pages/wrong_question_upload.py", title="错题录入", icon="📝")
diagnosis_page = st.Page("pages/diagnosis_generation.py", title="诊断与出题", icon="🧠")
export_page = st.Page("pages/practice_export.py", title="练习导出", icon="📄")

navigation = st.navigation(
    {
        "第一阶段功能": [student_page],
        "后续版本预览": [wrong_question_page, diagnosis_page, export_page],
    }
)
navigation.run()

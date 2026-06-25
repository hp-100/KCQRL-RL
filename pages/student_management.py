"""学生管理页面。"""

from __future__ import annotations

import streamlit as st

from src.database import DatabaseError, create_student, delete_student, get_student, list_students, update_student
from src.schemas import GRADE_OPTIONS, Student


def _current_student_label(student: Student | None) -> str:
    """生成当前学生展示文本。"""
    return f"当前学生：{student.name}（{student.grade}）" if student else "当前学生：未选择"


def _refresh_current_student() -> Student | None:
    """从 session_state 中读取并刷新当前学生。"""
    student_id = st.session_state.get("current_student_id")
    if student_id is None:
        return None
    try:
        student = get_student(int(student_id))
    except DatabaseError as exc:
        st.error(str(exc))
        return None
    if student is None:
        st.session_state.pop("current_student_id", None)
    return student


def _render_create_form() -> None:
    """渲染新增学生表单。"""
    st.subheader("新增学生")
    with st.form("create_student_form", clear_on_submit=True):
        name = st.text_input("姓名（必填）")
        grade = st.selectbox("年级", GRADE_OPTIONS)
        target = st.text_input("学习目标（可选）")
        notes = st.text_area("教师备注（可选）")
        submitted = st.form_submit_button("新增学生")

    if submitted:
        if not name.strip():
            st.warning("请输入学生姓名后再提交。")
            return
        try:
            student_id = create_student(name=name, grade=grade, target=target, notes=notes)
        except (ValueError, DatabaseError) as exc:
            st.error(str(exc))
            return
        st.session_state["current_student_id"] = student_id
        st.success("学生新增成功，并已设为当前学生。")
        st.rerun()


def _render_student_selector(students: list[Student]) -> None:
    """渲染当前学生选择器。"""
    st.subheader("选择当前学生")
    if not students:
        st.info("暂无学生，请先新增学生。")
        return

    id_to_student = {student.id: student for student in students}
    student_ids = list(id_to_student)
    current_id = st.session_state.get("current_student_id")
    default_index = student_ids.index(current_id) if current_id in student_ids else 0
    selected_id = st.selectbox(
        "当前学生",
        student_ids,
        index=default_index,
        format_func=lambda student_id: f"{id_to_student[student_id].name}（{id_to_student[student_id].grade}）",
    )
    st.session_state["current_student_id"] = selected_id
    st.info(_current_student_label(id_to_student[selected_id]))


def _render_student_editor(students: list[Student]) -> None:
    """渲染学生列表、编辑和删除功能。"""
    st.subheader("学生列表")
    if not students:
        st.info("暂无学生记录。")
        return

    for student in students:
        with st.expander(f"{student.name}（{student.grade}）", expanded=False):
            st.caption(f"创建时间：{student.created_at}｜更新时间：{student.updated_at}")
            with st.form(f"edit_student_{student.id}"):
                name = st.text_input("姓名（必填）", value=student.name, key=f"name_{student.id}")
                grade_index = GRADE_OPTIONS.index(student.grade)
                grade = st.selectbox("年级", GRADE_OPTIONS, index=grade_index, key=f"grade_{student.id}")
                target = st.text_input("学习目标（可选）", value=student.target, key=f"target_{student.id}")
                notes = st.text_area("教师备注（可选）", value=student.notes, key=f"notes_{student.id}")
                saved = st.form_submit_button("保存修改")

            if saved:
                if not name.strip():
                    st.warning("学生姓名不能为空。")
                else:
                    try:
                        updated = update_student(student.id, name=name, grade=grade, target=target, notes=notes)
                    except (ValueError, DatabaseError) as exc:
                        st.error(str(exc))
                    else:
                        st.success("学生资料已更新。" if updated else "未找到该学生，可能已被删除。")
                        st.rerun()

            confirm = st.checkbox("确认删除该学生", key=f"confirm_delete_{student.id}")
            if st.button("删除学生", key=f"delete_{student.id}", type="secondary"):
                if not confirm:
                    st.warning("删除前请先勾选“确认删除该学生”。")
                else:
                    try:
                        deleted = delete_student(student.id)
                    except DatabaseError as exc:
                        st.error(str(exc))
                    else:
                        if st.session_state.get("current_student_id") == student.id:
                            st.session_state.pop("current_student_id", None)
                        st.success("学生已删除。" if deleted else "未找到该学生，可能已被删除。")
                        st.rerun()


st.title("学生管理")
st.write("用于维护学生基础资料，并选择后续诊断与练习生成所使用的当前学生。")

current_student = _refresh_current_student()
st.metric("当前选择", current_student.name if current_student else "未选择")

try:
    student_list = list_students()
except DatabaseError as exc:
    st.error(str(exc))
    student_list = []

left, right = st.columns([1, 2])
with left:
    _render_create_form()
    _render_student_selector(student_list)
with right:
    _render_student_editor(student_list)

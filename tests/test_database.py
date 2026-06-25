"""数据库访问层基础测试。"""

from __future__ import annotations

from pathlib import Path

from src.database import create_student, delete_student, get_student, init_database, list_students, update_student


def test_student_crud_uses_temporary_database(tmp_path: Path) -> None:
    """验证新增、读取、修改、删除学生均使用临时数据库。"""
    db_path = tmp_path / "app.db"
    init_database(db_path)

    student_id = create_student("张三", "高三", "冲刺 130 分", "基础较好", db_path)
    student = get_student(student_id, db_path)

    assert student is not None
    assert student.name == "张三"
    assert student.grade == "高三"
    assert student.target == "冲刺 130 分"
    assert len(list_students(db_path)) == 1

    assert update_student(student_id, "李四", "复读", "稳定 120 分", "需要加强导数", db_path)
    updated_student = get_student(student_id, db_path)

    assert updated_student is not None
    assert updated_student.name == "李四"
    assert updated_student.grade == "复读"
    assert updated_student.notes == "需要加强导数"

    assert delete_student(student_id, db_path)
    assert get_student(student_id, db_path) is None
    assert list_students(db_path) == []

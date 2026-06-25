"""SQLite 数据库访问层。"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from src.schemas import GRADE_OPTIONS, Grade, Student

DEFAULT_DATABASE_PATH = Path("data/app.db")


class DatabaseError(RuntimeError):
    """数据库操作失败时抛出的友好异常。"""


def _resolve_database_path(db_path: str | Path | None = None) -> Path:
    """返回数据库路径，未指定时使用正式数据库。"""
    return Path(db_path) if db_path is not None else DEFAULT_DATABASE_PATH


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """创建 SQLite 连接，并启用外键约束。"""
    path = _resolve_database_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _row_to_student(row: sqlite3.Row) -> Student:
    """将数据库行转换为 Student 数据对象。"""
    return Student(
        id=int(row["id"]),
        name=str(row["name"]),
        grade=row["grade"],
        target=str(row["target"] or ""),
        notes=str(row["notes"] or ""),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def init_database(db_path: str | Path | None = None) -> None:
    """初始化数据库目录、数据库文件和 students 表。"""
    try:
        with _connect(db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    grade TEXT NOT NULL CHECK (grade IN ('高一', '高二', '高三', '复读')),
                    target TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                )
                """
            )
    except sqlite3.Error as exc:
        raise DatabaseError("初始化数据库失败，请检查 data 目录权限。") from exc


def create_student(name: str, grade: Grade, target: str = "", notes: str = "", db_path: str | Path | None = None) -> int:
    """新增学生并返回学生 ID。"""
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("学生姓名不能为空。")
    if grade not in GRADE_OPTIONS:
        raise ValueError("年级必须为：高一、高二、高三或复读。")
    try:
        with _connect(db_path) as connection:
            cursor = connection.execute(
                "INSERT INTO students (name, grade, target, notes) VALUES (?, ?, ?, ?)",
                (clean_name, grade, target.strip(), notes.strip()),
            )
            return int(cursor.lastrowid)
    except sqlite3.Error as exc:
        raise DatabaseError("新增学生失败，请稍后重试。") from exc


def list_students(db_path: str | Path | None = None) -> list[Student]:
    """按更新时间倒序返回学生列表。"""
    try:
        with _connect(db_path) as connection:
            rows: Iterable[sqlite3.Row] = connection.execute(
                """
                SELECT id, name, grade, target, notes, created_at, updated_at
                FROM students
                ORDER BY datetime(updated_at) DESC, id DESC
                """
            ).fetchall()
            return [_row_to_student(row) for row in rows]
    except sqlite3.Error as exc:
        raise DatabaseError("读取学生列表失败，请稍后重试。") from exc


def get_student(student_id: int, db_path: str | Path | None = None) -> Student | None:
    """根据 ID 获取单个学生，未找到时返回 None。"""
    try:
        with _connect(db_path) as connection:
            row = connection.execute(
                "SELECT id, name, grade, target, notes, created_at, updated_at FROM students WHERE id = ?",
                (student_id,),
            ).fetchone()
            return _row_to_student(row) if row is not None else None
    except sqlite3.Error as exc:
        raise DatabaseError("读取学生资料失败，请稍后重试。") from exc


def update_student(student_id: int, name: str, grade: Grade, target: str = "", notes: str = "", db_path: str | Path | None = None) -> bool:
    """更新学生资料，成功更新返回 True，未找到学生返回 False。"""
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("学生姓名不能为空。")
    if grade not in GRADE_OPTIONS:
        raise ValueError("年级必须为：高一、高二、高三或复读。")
    try:
        with _connect(db_path) as connection:
            cursor = connection.execute(
                """
                UPDATE students
                SET name = ?, grade = ?, target = ?, notes = ?, updated_at = datetime('now', 'localtime')
                WHERE id = ?
                """,
                (clean_name, grade, target.strip(), notes.strip(), student_id),
            )
            return cursor.rowcount > 0
    except sqlite3.Error as exc:
        raise DatabaseError("修改学生资料失败，请稍后重试。") from exc


def delete_student(student_id: int, db_path: str | Path | None = None) -> bool:
    """删除学生，成功删除返回 True，未找到学生返回 False。"""
    try:
        with _connect(db_path) as connection:
            cursor = connection.execute("DELETE FROM students WHERE id = ?", (student_id,))
            return cursor.rowcount > 0
    except sqlite3.Error as exc:
        raise DatabaseError("删除学生失败，请稍后重试。") from exc

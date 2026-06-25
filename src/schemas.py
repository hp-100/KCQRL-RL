"""应用数据结构定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Grade = Literal["高一", "高二", "高三", "复读"]
GRADE_OPTIONS: tuple[Grade, ...] = ("高一", "高二", "高三", "复读")


@dataclass(frozen=True)
class Student:
    """学生资料。"""

    id: int
    name: str
    grade: Grade
    target: str
    notes: str
    created_at: str
    updated_at: str

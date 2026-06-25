# 高考数学错题诊断与个性化变式题生成系统

这是一个基于 Python 和 Streamlit 的高考数学错题管理应用。第一阶段聚焦学生基础资料管理，为后续错题录入、知识点诊断、个性化变式题生成和练习导出打基础。本阶段不接入任何大语言模型 API，不包含图片识别或 Word 导出功能。

## Python 环境要求

- Python 3.10 或更高版本
- macOS 本地环境可直接运行
- SQLite 使用 Python 标准库 `sqlite3`，无需单独安装数据库服务

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行命令

```bash
streamlit run app.py
```

首次运行时，程序会自动创建 `data/app.db` 和 `students` 表。学生数据会持久化保存在本地 SQLite 数据库中，关闭并重新启动程序后仍然存在。

## 当前已经实现的功能

- 使用 `st.navigation` 构建四个中文页面：学生管理、错题录入、诊断与出题、练习导出。
- 学生管理页面已实现：
  - 新增学生；
  - 显示学生列表；
  - 修改学生资料；
  - 删除学生，并要求删除前再次确认；
  - 选择当前学生；
  - 使用 `st.session_state` 保存当前选择的学生；
  - 页面显示当前学生名称；
  - 输入为空时给出明确提示；
  - 数据库操作失败时显示友好错误信息。
- 错题录入、诊断与出题、练习导出页面目前为占位说明页面。
- 数据库访问代码已拆分到 `src/database.py`，并使用参数化 SQL。
- 数据库基础增删改查测试使用临时数据库，不会修改正式的 `data/app.db`。

## 如何运行测试

```bash
pytest tests/test_database.py
```

## 项目结构

```text
app.py
requirements.txt
README.md
.gitignore
pages/student_management.py
pages/wrong_question_upload.py
pages/diagnosis_generation.py
pages/practice_export.py
src/__init__.py
src/database.py
src/schemas.py
data/.gitkeep
data/uploads/questions/.gitkeep
data/uploads/student_solutions/.gitkeep
tests/test_database.py
```

## 后续计划

1. 实现错题录入：支持题目文本、题目来源、知识点标签和学生解答记录。
2. 实现诊断与出题：在接入模型前先设计可解释的规则化诊断流程。
3. 实现练习导出：先支持页面预览和可复制文本，再考虑文件导出。
4. 补充更多测试：包括表单校验、数据库异常处理和页面冒烟测试。

## 常见问题

### 运行后找不到数据库怎么办？

程序会在首次启动时自动创建 `data/app.db`。如果创建失败，请检查项目目录权限，确认当前用户可以写入 `data` 目录。

### 测试会不会污染正式数据库？

不会。`tests/test_database.py` 使用 pytest 的 `tmp_path` 创建临时 SQLite 数据库，不会读写 `data/app.db`。

### 是否需要配置 API Key？

不需要。第一阶段不接入任何大语言模型 API，代码中也不会保存 API 密钥。

### 为什么其他三个页面还不能使用？

本阶段只完整实现学生管理。错题录入、诊断与出题、练习导出将在后续版本逐步实现。

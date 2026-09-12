# Pipewise 项目功能

## 1. 项目定位

`Pipewise` 是一个围绕 `pandas.DataFrame` 构建的轻量数据处理管道库。它不替代
`pandas`，而是把「按列声明输入、按步骤串联处理、自动回写结果列」封装成统一的
注册式流水线，减少业务代码里重复的列选择、逐行处理和中间结果管理。

当前版本 `2.0.0`，依赖仅 `pandas` 与 `tqdm`。

## 2. 主要功能

### 2.1 任务注册与串行执行

- 通过 `@pipewise.register(...)` 注册任务，支持裸装饰器与带参装饰器。
- 函数中无默认值的参数名自动映射为 DataFrame 同名列；`**kwargs` 接收其余所有列。
- 任务按注册顺序执行，后续任务可直接依赖前序任务新增的列。
- 注册时解析一次函数签名并缓存到 `TaskDef`，执行期不再重复解析。

**注意**：带默认值的参数不会从 DataFrame 读取。若存在同名输入列，注册时会发出
告警说明该列将被忽略。

### 2.2 多种输出模式

`register(outputs=...)` 支持：

- `None`：仅执行副作用，不写回。
- `"col"`：单列输出。
- `["c1", "c2"]`：固定多列输出。
- `{"col": type}`：带类型声明，执行后自动 `astype`。
- `"dict"`：动态字典输出，可按行生成不同列。

### 2.3 向量化优先，逐行回退需显式开启

- 默认 `vectorized=True`，整列 `Series` 直接传给函数。
- 若向量化调用抛出 `TypeError` / `ValueError` / `AttributeError`，且注册时显式传入
  `fallback_on_vectorized_error=True`，则回退为逐行执行。
- **2.0 起默认关闭自动回退**：向量化调用已经执行过一次函数体，回退会再次执行，
  可能重复副作用。默认行为是直接抛出错误，并给出「改用 `vectorized=False`」的提示。
- 逐行执行使用 `DataFrame.itertuples`，实测比 `DataFrame.apply(axis=1)` 快约 5 倍，
  行内值为 Python 原生标量。

### 2.4 AST 向量化兼容检测

注册时扫描函数源码，对**真正的输入列参数**告警以下模式：

- `len(col)`、`isinstance(col, ...)`、`type(col)`
- `col.split(...)`、`col.strip(...)`、`col.lower()`、`col.upper()`
- `col[0]`、`col['key']`
- 用列作为分支条件：`if col > 0:`、`if col:`、`x if col else y`

检测只针对函数参数名，局部变量与辅助对象不会误报。

### 2.5 GroupBy 分组执行

`groupby="col"` 或 `["c1", "c2"]`，走 split-apply-combine，每个分组独立执行同一函数
再合并回原表。

### 2.6 Schema 校验

支持 `dtype`、`nullable`、`allowed_values`、`min`、`max` 五种规则键，可声明在
管道输入、任务输入、任务输出三处。校验时每列只计算一次 `dropna()` 结果并复用。

### 2.7 失败回滚

`run()` 执行前保存快照；任一任务失败即回滚本次执行期间的所有列改动，统一抛出
`PipewiseExecutionError`，原始异常保留在 `__cause__`。

### 2.8 任务管理与可观测性

- `tasks`：返回 `TaskSummary`（`func_name` / `outputs` / `groupby` / `vectorized`）。
- `remove(func)` / `clear()`：增删任务。
- `plan()`：通过 `logging.info` 输出执行计划。
- `run(task="name")`：仅执行唯一匹配的单个任务，便于调试。

### 2.9 非常规数据结构的兼容

列中可以放非标量值，两种执行模式均支持：`list`、`dict`、`set`、`frozenset`、
`tuple`、`numpy.ndarray`、类对象、类实例、嵌套容器。

向量化函数还可返回：

- 二维 `numpy.ndarray` → 多列输出；
- `(n, 1)` 数组或单列 `DataFrame`/`Series` → 单列输出。

输出长度/形状不匹配时抛出带函数名与列名的 `PipewiseOutputAssignmentError`，
而不是让 pandas 报晦涩错误或静默广播。

## 3. 代码结构

| 模块 | 职责 |
|---|---|
| `__init__.py` | 公共接口、版本与作者 |
| `core.py` | `Pipewise` 门面：注册、`run`、任务管理、编排、回滚 |
| `_execution.py` | 向量化 / 逐行 / 分组三种执行策略，以及回退判定 |
| `_assign.py` | 输出回写、索引对齐、长度与形状守卫 |
| `_tasks.py` | `TaskDef` 元数据、签名解析、注册参数归一化 |
| `_schema.py` | schema 归一化与校验、dtype 匹配、类型转换 |
| `_hazards.py` | 基于 AST 的向量化危险模式检测 |
| `errors.py` | 异常层次 |

设计要点：向量化的「调用」与「回写」严格分离，只有调用阶段的失败才可能触发回退，
回写阶段的问题一律直接暴露，避免被误判为向量化不兼容而静默产生错误结果。

## 4. 测试

`tests/test_pipewise.py` 覆盖：

1. 向量化多列输出、动态字典输出、类型声明输出。
2. 逐行与分组执行，以及显式开启的回退行为。
3. 管道输入 / 任务输入 / 任务输出 schema 校验。
4. `inplace` 语义与失败回滚。
5. 任务管理：`tasks` / `remove` / `clear` / `plan` / 单任务执行。
6. 非常规数据结构：list / dict / set / frozenset / tuple / `numpy.ndarray` /
   类对象 / 类实例 / 嵌套容器。
7. 索引对齐、dtype 与空值边界（非连续索引、全空列、空表、单行表）。
8. 输出形状守卫：长度不匹配、参差返回值、非法回写。
9. AST 危险检测的命中与误报抑制。
10. 默认值参数吞掉同名列的告警。

## 5. 结论

`Pipewise` 把 DataFrame 处理组织成可注册、可串联、可回滚的任务流水线，并在
易用性与性能之间取得平衡。2.0 的目标是**让隐式行为显式化**：回退需显式开启、
危险模式在注册时提示、形状错误明确报错，同时通过 `itertuples`、单次 `dropna`
与模块拆分提升性能与可维护性。

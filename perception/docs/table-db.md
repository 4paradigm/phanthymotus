# `table_db` 接口说明

`table_db` 是基于 SQLite 的通用数据表卡片：一个库里放多张表，表是最大一级，没有「数据库」这一层。
动作和参数尽量贴近 SQL 原语，因为调用它的主要是 LLM：写法越像它已经熟悉的 SQL，越少出错。

## 动作

| action | SQL 对应 | 参数（粗体必填） | 返回 |
|---|---|---|---|
| `list_tables` | `.tables` | 无 | `tables: [{table, rows}]` |
| `describe` | `.schema` | **`table`** | `columns: [{name, type}]`, `rows` |
| `create_table` | `CREATE TABLE` | **`table`**, **`columns`** | `columns` |
| `add_column` | `ALTER TABLE … ADD COLUMN` | **`table`**, **`columns`** | 加列后的 `columns` |
| `drop_table` | `DROP TABLE` | **`table`**, **`confirm: true`** | `dropped`, `rows` |
| `insert` | `INSERT INTO` | **`table`**, **`values`**, `file` | `inserted`, `rowids` |
| `select` | `SELECT` | **`table`**, `fields`, `where`, `group_by`, `order_by`, `limit`, `offset` | `columns`, `total`, `has_more`, `rows` |
| `update` | `UPDATE … SET … WHERE` | **`table`**, **`where`**, `set`, `file` | `updated`, `rowids`, `rows` |
| `delete` | `DELETE FROM … WHERE` | **`table`**, **`where`** | `deleted`, `rowids` |
| `import` | `.import` | **`table`**, **`path`**, `mode`, `header_row` | `imported`, `blobs`, `columns`, `rowids` |
| `export` | 导出 | **`table`**, `format`, `fields`, `where`, `order_by` | `path`, `url` |

`update` 和 `delete` 的 `where` 必填：改全表要显式写 `[["rowid", ">", 0]]`。`drop_table` 不可恢复，必须带 `confirm: true`。

## 参数写法

| 参数 | 写法 | 例 |
|---|---|---|
| `columns` | 列定义，`type` 为 `TEXT`（默认）/ `INTEGER` / `REAL` / `BLOB` | `[{"name": "姓名"}, {"name": "金额", "type": "REAL"}]` |
| `fields` | 要返回的列和聚合：`COUNT(*)`、`COUNT(列)`、`COUNT(DISTINCT 列)`、`SUM`、`AVG`、`MIN`、`MAX` | `["类别", "SUM(金额)", "COUNT(*)"]` |
| `where` | 条件数组，AND 连接；每个条件 `[列, 运算符, 值]` | `[["金额", ">", 100], ["姓名", "like", "%张%"]]` |
| `group_by` | 分组列；`fields` 里的普通列必须在这里 | `["类别"]` |
| `order_by` | `列名` 或 `列名 DESC`；分组查询可按聚合排序 | `["SUM(金额) DESC"]` |
| `values` | 行对象数组 | `[{"姓名": "张三", "金额": 12.5}]` |
| `set` | 新值；`null` 清空 | `{"状态": "已完成"}` |

`where` 运算符：`=` `!=` `>` `>=` `<` `<=` `like` `not like` `in` `not in`（值为数组）`between`（值为 `[下限, 上限]`）`is null` `is not null`（不写值）。行号列叫 `rowid`。
旧的对象写法 `{"列": 值}` 仍然接受，表示等值（数组值表示 IN，`null` 表示 IS NULL）。

带聚合或 `group_by` 的查询每组返回一行，键名就是 `fields` 里写的那一项，`total` 是组数。和 SQL 一样，没有聚合的列必须出现在 `group_by` 里。

## 文件 / BLOB

BLOB 列存图片或任意文件，字节存在库里，删行、删表、替换、清空时自动回收。

### 文件怎么到卡片

perception 和 agent-core 是两个容器，不共享文件系统，所以文件要**先传到本卡片**，用的是卡片自己的
`/file/upload`（`utils/file_intake.py`），落在 `/models/uploads/<日期>/` 下，7 天后清理——写入 BLOB 时字节已拷进库里，清理不影响数据。
两条入口都走这里：

- **画布**：`file`（BLOB）和 `path`（导入表格）两个参数是「选择文件」按钮，浏览器经 agent-core 转发上传，
  按钮把卡片这边的路径填回参数。
- **对话**：用户在飞书 / QQ 等渠道发来的附件，先由 agent-core 存进它自己的容器。LLM 把这个路径填进
  `file` / `path` 参数；这两个参数声明了 `format: "file"`，agent-core 调用前会自动把文件传到本卡片、换成卡片这边的路径。

**只有 `file` / `path` 会被自动传送。** 写在 `values` / `set` 里的路径原样交给卡片，指向 agent-core 容器里的文件就会「文件不存在」。
所以 BLOB 列一律写 `"$file"`，文件路径放 `file`；每次调用带一个文件。

| 要做的事 | 写法 |
|---|---|
| 新增一行带文件 | `insert`：`values: [{"姓名": "张三", "照片": "$file"}]`，`file: "<路径>"` |
| 给已有行加 / 换文件 | `update`：`set: {"照片": "$file"}`，`file: "<路径>"`，`where: [["rowid", "=", 3]]` |
| 删掉某行的文件、保留这一行 | `update`：`set: {"照片": null}` |
| 删行连同文件 | `delete` |
| 给表加一个文件列 | `add_column`：`[{"name": "合同", "type": "BLOB"}]` |
| 读文件 | `select` 返回 `{"type": "BLOB", "name", "mime", "size", "url"}`，`url` 直接 HTTP GET 下载 |
| 找有 / 没有文件的行 | `where`：`[["照片", "is not null"]]` / `[["照片", "is null"]]` |

表里只有一个 BLOB 列时，`values` / `set` 里可以不写 `"$file"`，`file` 自动写入那一列。
卡片自己容器里已有的文件（`/models`、`/tmp`、`/work` 下）也可以直接写路径，或写 `{"readfile": "路径"}`。
配置了 `ACCESS_TOKEN` 时，下载链接自动带 24 小时有效的签名。

`import` / `export` 只支持 `.csv` 和 `.xlsx`。`.xlsx` 导入时单元格里的图片存为 BLOB 列，导出 `.xlsx` 时图片嵌回单元格；导出 `.csv` 时 BLOB 列写下载链接。

## 只读与写操作（peer 权限）

`list_tables`、`describe`、`select`、`export` 在 `x-action-params` 里声明 `readOnly: true`，其余为写操作。
perception 层默认被当作只读、`viewer` 级别的 peer 可以直接调用；这张卡用 `readOnly` 声明把写操作排除在外，
viewer 只能读，写和删要 `operator`。

## 错误返回

失败统一返回 `{"ok": false, "reason": "...", "detail": "...", "written": 0}`，写操作整批回滚。
`detail` 会说明正确写法；表名、列名写错时列出现有的表和列。
常见 `reason`：`bad_request`、`unknown_table`、`table_exists`、`unknown_column`、`column_exists`、
`not_found`、`invalid_header`、`invalid_value`、`unsupported_format`、`unreadable`、`empty`、`too_large`、`busy`、`internal_error`。

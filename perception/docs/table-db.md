# `table_db` 接口说明

`table_db` 是基于 SQLite 的通用数据表卡片。单库多表（表为最大一级，无数据库概念），对外接口严格对应 SQLite 原语，不耦合任何具体业务卡片。

## 动作一览

| action | SQLite 对应 | 参数（粗体必填） | 返回重点 |
|---|---|---|---|
| `tables` | `.tables` / `.schema` | 无 | `count`, `table_names`, `tables`（含列名、类型、行数） |
| `create_table` | `CREATE TABLE` | **`table`**, **`columns`** | `table`, `columns` |
| `drop_table` | `DROP TABLE` | **`table`** | `table`, `dropped: true`, `rows` |
| `insert` | `INSERT INTO` | **`table`**, **`values`**, `file` | `inserted`, `rowids` |
| `select` | `SELECT` | **`table`**, `columns`, `where`, `group_by`, `order_by`, `limit`, `offset` | `total`, `has_more`, `rows`（或聚合结果） |
| `update` | `UPDATE` | **`table`**, **`where`**, `set`, `file` | `updated`, `rowids`, `rows` |
| `delete` | `DELETE FROM` | **`table`**, **`where`** | `deleted`, `rowids` |
| `import` | `.import` | **`table`**, **`path`**, `mode`, `header_row` | `imported`, `blobs`, `columns`, `rowids` |
| `export` | 导出表格 | **`table`**, `format`, `where`, `columns`, `order_by` | `format`, `rows`, `path`, `url` |

## 列类型（SQLite 标准）

- `TEXT`（默认）
- `INTEGER`
- `REAL`
- `BLOB`（二进制文件）

## 文件 / BLOB 存取方式

- **写入（SQLite `readfile` 语义）**：在 `insert` 的 `values` 或 `update` 的 `set` 中，为 `BLOB` 列传入文件路径（或 `{"readfile": "路径"}`）；画布上可通过 `file` 选择器上传并在 `values`/`set` 中写 `'$file'`（表仅有一个 `BLOB` 列时可自动绑定）。将 `BLOB` 列更新为 `null` 即清空并回收存储空间。
- **读取**：`select` 返回的非空 `BLOB` 列包含 `{"type": "BLOB", "name", "mime", "size", "url"}`，可直接通过 HTTP GET 下载原始字节（配置 `ACCESS_TOKEN` 时自动签发带过期时间的 `?exp=&sig=` 链接）。
- **整表导入导出**：`import` 与 `export` 均仅支持 `.csv` 和 `.xlsx`。`.xlsx` 导入时自动提取单元格图片存为 `BLOB` 列，导出为 `.xlsx` 时自动将图片 `BLOB` 嵌入回单元格。

## 错误返回

所有失败统一返回 `{"ok": false, "reason": "...", "detail": "...", "written": 0}`，写操作整批事务回滚。常见 `reason`：`bad_request`、`unknown_table`、`table_exists`、`unknown_column`、`not_found`、`invalid_header`、`invalid_value`、`unsupported_format`、`unreadable`、`empty`、`too_large`、`busy`、`internal_error`。

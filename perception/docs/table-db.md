# `table_db`：给技能作者的使用说明

一张给所有 skill 共用的**表格数据库卡片**。你的 skill 指定 `db`（库名，用你的 skill slug）和 `table`（表名），就能：

- 把用户在飞书发来的 Excel / CSV（兼容 Word 里的表格，单元格可以有图片）导入成一张表；
- 在运行中**新增、查询、修改、删除、计数**；
- 把表里的照片交给人脸 / OCR / 物体识别卡片。

编号（`_id`）和计数由卡片保证。**不要再让模型用 Read/Write 改写 JSON 文件**。

---

## 一、接入三步

1. 技能的依赖工具（`requiredTools`）加 `"table_db"`；要录人脸再加 `"face_recognition"`。
2. 画布上把**决策卡**底部的「连接执行器」连到 `table_db` 卡。不连的话模型看不到这张卡。
3. 在指令里写清楚你的 `db` 和要用的表（模板见第四节）。

---

## 二、数据长什么样

- 列只有两种类型：
  - `text`：文字；
  - `file`：非文字，图片、PDF、音频都行，单个文件不超过 8 MiB。
- 一条记录：

```json
{"_id": 7, "_created_at": "2026-11-20 10:20:31", "_updated_at": "2026-11-20 10:20:31",
 "姓名": "张三", "准考证号": "0012",
 "照片": {"file_id": "f_42", "name": "张三.jpg", "mime": "image/jpeg", "size": 18432}}
```

  - `_id` 是卡片生成的编号，删除后不会复用。
  - 文字永远是字符串，所以 `0012` 的前导零不会丢。空值不出现在记录里。
  - 文件列只给出文件信息。要拿到文件本身，用 `get_file`。

- 失败的统一格式（`written` 永远是 0，因为整批回滚）：

```json
{"ok": false, "reason": "invalid_header", "detail": "第 1 行表头「姓名」重复（A 列和 C 列）",
 "row": 1, "column": "姓名", "written": 0}
```

---

## 三、13 个动作

除 `list`、`drop_db` 外，每个动作都要传 `table`；`db` 可选，默认 `default`，**建议一律传你的 skill slug**。

**库什么时候会被新建**：只有 `create`，以及 `import` 的 `mode=create`，在 `db` 填一个还没有的名字时会新建这个库（返回 `db_created: true`）。其它所有动作，`db` 必须是已有的库，否则报 `unknown_db`，并列出现有的库，不会因为库名打错而多出一个空库。一次失败的首次导入或建表也不会留下空库。

| 动作 | 参数 | 返回 |
|---|---|---|
| `list` | （不填） | 先给汇总：`databases_count`、`databases_list`、`tables_list`（每个库的表数和表名）；然后是 `databases`：每张表的列类型、行数、文件数。**不知道库名、表名时先调它** |
| `create` | `columns`：`[{"name":"姓名","type":"text"},{"name":"照片","type":"file"}]` | 列定义 |
| `import` | `path`：飞书文件直接填 `files[].path`<br>`mode`：`create`（默认，建新表；表已存在报 `table_exists`）或 `append`（追加到已有的表；库不存在报 `unknown_db`，表不存在报 `unknown_table`；不去重）<br>`header_row`：表头在第几行（自动识别错了才填）<br>`dry_run`：`true` 表示只检查不写<br> | `columns`、`rows`、`imported`、`files`、`skipped_rows`（空行），以及 3 条 `sample` |
| `append` | `records`：`[{...}]`（不超过 500 条，只写文字列）<br> | `ids` |
| `query` | `where`、`offset`、`limit`（默认 20，最多 200） | `total`、`count`、`has_more`、`next_offset`、`records` |
| `update` | `_id`、`set`：`{"状态":"已归还"}`，值为 `null` 表示清空 | 修改后的完整记录 |
| `delete` | `_id` | `deleted`（软删除，查询和统计会自动跳过） |
| `stats` | `where`、`group_by` | `total`，以及 `groups: [{"value","count"}]` |
| `attach_file` | `_id`、`column`（文件列）、`file` | 文件信息（会替换原来的文件） |
| `get_file` | `file_id` | `path`：本容器里的临时文件，24 小时有效 |
| `export_files` | `column`（文件列）、`name_column`、`profile_columns`、`where` | `dir`：照片加 `manifest.json`，交给 `face_recognition.register_by_corpus` |
| `drop_table` | `confirm`：再填一遍表名 | 删除整张表及其文件（不可恢复），用于重新导入 |
| `drop_db` | `db`（必填）、`confirm`：再填一遍库名 | 删除整个库：库里所有表、记录、文件一起删除（不可恢复），返回删掉的表名、行数、文件数 |

**图片的三种放法都支持**：WPS 的嵌入单元格图片（`DISPIMG`）、浮动图片（按锚点归到对应行）、Excel 365 的「放在单元格中」图片（rich data）。
超过 15 位、却按数字存储的号码（常见于身份证号），Excel 本身只保存了 15 位有效数字，后面几位已经丢失；导入时会在 `warnings` 里提醒，请用户把这一列改为文本格式后重新填写。

**在画布上直接测试**：卡片上只有一个 action 下拉框、各个参数格子和一个「▶ 执行」按钮。
`where`、`set`、`records` 填 JSON 文本，**单引号和双引号都可以**，例如 `{'姓名':'张三'}`。画布的灰色提示里用的是单引号，因为画布显示提示时不转义双引号，双引号会把提示截断。`columns` 和 `profile_columns` 可以写成 `姓名,照片:file` 这样的逗号简写，中文标点也能识别。
漏填必填项时，会返回 `bad_request` 并带上 `missing`。

**`where` 只有三种写法**：
- `{"列":"值"}`：等于；
- `{"列":["a","b"]}`：等于其中之一；
- `{"列":null}`：为空。

多个键之间是「并且」，也可以用 `"_id"` 作为键。其它写法会报 `bad_request`。

---

## 四、指令模板（直接抄）

### 名单检录（导入 + 录人脸 + 签到）

```text
数据一律用 table_db，db="q5-exam-helper"；不要用 Bash，也不要读写 JSON 文件。
【导入名单】用户发来名单文件：
1. table_db.import(table="名单", path=消息里的 files[0].path, dry_run=true)
   把 columns、rows、files（照片数）报给用户，等用户说「确认」。
2. 用户确认后，同样的参数去掉 dry_run，正式导入。
3. 失败时把 detail 原话告诉用户（里面有第几行、哪一列）。
【录人脸】用户说「录入人脸」：
4. table_db.export_files(table="名单", column="照片", name_column="姓名", profile_columns=["准考证号"])
5. face_recognition.register_by_corpus(package=上一步的 dir)，逐人汇报结果。
【签到】有人签到：table_db.append(table="签到", records=[{"准考证号":..,"姓名":..,"状态":"已到"}])
【统计】「到了几个人」用 table_db.stats(table="签到", group_by="状态")，不要自己数。
```

### 借还台账（运行时维护）

```text
台账用 table_db，db="g1-borrow-keeper"，table="借还"。
第一次使用时，如果 query 报 unknown_table，先 create：
  columns=[{"name":"借用人","type":"text"},{"name":"物品","type":"text"},{"name":"状态","type":"text"},
           {"name":"借出时间","type":"text"},{"name":"归还时间","type":"text"},{"name":"照片","type":"file"}]
- 借出：append(records=[{"借用人":..,"物品":..,"状态":"借出","借出时间":当前时间}])，把返回的 _id 当作编号念给用户。
- 拍照留档：vision_capture 拍照后调用 attach_file(_id=该编号, column="照片", file=照片路径)。
- 归还：query(where={"物品":..,"状态":"借出"}) 找到那一条，再 update(_id, set={"状态":"已归还","归还时间":当前时间})。
- 谁还没还：stats(where={"状态":"借出"}, group_by="借用人")。
```

### 表里的图片做 OCR / 识别物体

```text
query(table=.., where=..) 拿到 照片.file_id → get_file(file_id) 得到 path
→ ocr.recognize_by_photo(image_path=path) 或 vop.recognize_by_photo(image_path=path)
→ 需要留档就 update(_id, set={"识别结果": ...})
```

---

## 五、错误码

| reason | 什么时候出现 | skill 应该怎么做 |
|---|---|---|
| `bad_request` | 参数缺失或类型不对，`where` 写法不支持，往文件列写文字 | 按 `detail` 改参数 |
| `unknown_db` | 库不存在（只有 create / import 的 create 模式会新建库） | 用 `list` 看有哪些库，或检查 `db` 有没有打错 |
| `unknown_table` / `table_exists` | 表不存在 / 建表时表已存在 | 先 create 或 import；追加用 `mode="append"` |
| `unknown_column` | 用了表里没有的列 | `detail` 里列出了现有的列 |
| `not_found` | `_id` 或 `file_id` 不存在，或已删除 | 先 query 确认 |
| `invalid_header` | 表头重复，或找不到表头 | 请用户改表，或传 `header_row` |
| `invalid_value` | 图片列里出现文字，或类型对不上 | 把 `detail`（含行号和列名）告诉用户 |
| `legacy_format` / `unsupported_format` | .xls/.doc 等旧格式 / 不是表格 | 请用户另存为 xlsx 或 csv |
| `unreadable` / `empty` | 文件损坏 / 没有数据 | 请用户重新发送 |
| `too_large` | 超出上限（单个文件 8 MiB、2 万行、64 列、单格 2000 字、单库 2 GiB） | `detail` 会说明超了哪一项 |
| `request_conflict` | 同一个 用在了不同的内容上 | 换一个新的 |
| `busy` | 数据库正在被别的写操作占用 | 稍后重试 |
| `internal_error` | 意外错误（已记录日志，卡片不会崩溃） | 告诉用户稍后再试 |

---

## 六、边界

- **管**：导入、存储、增删改查、计数，以及把文件交出去（`get_file` / `export_files`）。
- **不管**：业务含义（什么算逾期、谁算已签到）、调用人脸 / OCR / VOP、发飞书。
- **录人脸会写库**，属于有副作用的操作，一定要在用户确认之后再调用 `register_by_corpus`。
  人脸卡会按相似度决定是合并到已有的人还是新建一个人，所以录完要汇报返回的 `person_id`。
- `get_file` 返回的 `path` 在 perception 容器里，视觉卡可以直接读；**agent-core 的 Read 读不到**（它只能读自己容器的 /work、/tmp）。模型要自己看文件时，用 Bash 执行返回里的 `agent_core.download`（curl 下载到 /tmp），再 Read `agent_core.local_path`。
- 数据保存在 `/models/table_db/<db>.db`：重启和重新部署都不会丢；一个库就是一个文件，复制即备份。

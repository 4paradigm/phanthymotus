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

## 三、10 个动作

每个动作都要传 `table`；`db` 可选，默认 `default`，**建议一律传你的 skill slug**。

| 动作 | 参数 | 返回 |
|---|---|---|
| `create` | `columns`：`[{"name":"姓名","type":"text"},{"name":"照片","type":"file"}]` | 列定义 |
| `import` | `path`：飞书文件直接填 `files[].path`<br>`mode`：`create`（默认）或 `append`<br>`header_row`：表头在第几行（自动识别错了才填）<br>`dry_run`：`true` 表示只检查不写<br>`request_id`：重试安全 | `columns`、`rows`、`imported`、`files`、`skipped_rows`（空行），以及 3 条 `sample` |
| `append` | `records`：`[{...}]`（不超过 500 条，只写文字列）<br>`request_id` | `ids` |
| `query` | `where`、`offset`、`limit`（默认 20，最多 200） | `total`、`count`、`has_more`、`next_offset`、`records` |
| `update` | `_id`、`set`：`{"状态":"已归还"}`，值为 `null` 表示清空 | 修改后的完整记录 |
| `delete` | `_id` | `deleted`（软删除，查询和统计会自动跳过） |
| `stats` | `where`、`group_by` | `total`，以及 `groups: [{"value","count"}]` |
| `attach_file` | `_id`、`column`（文件列）、`path` | 文件信息（会替换原来的文件） |
| `get_file` | `file_id` | `path`：本容器里的临时文件，24 小时有效 |
| `export_files` | `column`（文件列）、`name_column`、`profile_columns`、`where` | `dir`：照片加 `manifest.json`，交给 `face_recognition.register_by_corpus` |

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
2. 用户确认后，同样的参数去掉 dry_run，加 request_id="名单-<文件名>"，正式导入。
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
- 拍照留档：vision_capture 拍照后调用 attach_file(_id=该编号, column="照片", path=照片路径)。
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
| `unknown_table` / `table_exists` | 表不存在 / 建表时表已存在 | 先 create 或 import；追加用 `mode="append"` |
| `unknown_column` | 用了表里没有的列 | `detail` 里列出了现有的列 |
| `not_found` | `_id` 或 `file_id` 不存在，或已删除 | 先 query 确认 |
| `invalid_header` | 表头重复，或找不到表头 | 请用户改表，或传 `header_row` |
| `invalid_value` | 图片列里出现文字，或类型对不上 | 把 `detail`（含行号和列名）告诉用户 |
| `legacy_format` / `unsupported_format` | .xls/.doc 等旧格式 / 不是表格 | 请用户另存为 xlsx 或 csv |
| `unreadable` / `empty` | 文件损坏 / 没有数据 | 请用户重新发送 |
| `too_large` | 超出上限（单个文件 8 MiB、2 万行、64 列、单格 2000 字、单库 2 GiB） | `detail` 会说明超了哪一项 |
| `request_conflict` | 同一个 `request_id` 用在了不同的内容上 | 换一个新的 `request_id` |
| `busy` | 数据库正在被别的写操作占用 | 稍后重试 |
| `internal_error` | 意外错误（已记录日志，卡片不会崩溃） | 告诉用户稍后再试 |

---

## 六、边界

- **管**：导入、存储、增删改查、计数，以及把文件交出去（`get_file` / `export_files`）。
- **不管**：业务含义（什么算逾期、谁算已签到）、调用人脸 / OCR / VOP、发飞书。
- **录人脸会写库**，属于有副作用的操作，一定要在用户确认之后再调用 `register_by_corpus`。
  人脸卡会按相似度决定是合并到已有的人还是新建一个人，所以录完要汇报返回的 `person_id`。
- `get_file` 返回的路径在 perception 容器里，视觉卡可以读，但 **agent-core 的 Read 和飞书回复读不到**。不要尝试把文件发回飞书。
- 数据保存在 `/models/table_db/<db>.db`：重启和重新部署都不会丢；一个库就是一个文件，复制即备份。

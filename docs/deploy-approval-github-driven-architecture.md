# Deploy Approval GitHub 驱动架构说明

本文是 Deploy Approval 的中文主文档，描述当前冻结的 stateless 合同。

Deploy Controller 只负责部署审批与状态编排，**代码不修改**；代码构建与 Review 仍由 Review Agent 负责。

生产环境能力与仓库合同：
- 源/产品能力支持的仓库：`4paradigm/phanthymotus`、`4paradigm/phanthymotus-driver`
- 当前生产发布 / 默认 `GITHUB_REPOS` 为 `4paradigm/phanthymotus`（main）单仓库
- 当前 GitHub App 安装可能仅授权了 main
- `4paradigm/phanthymotus-driver` 运行时授权延后（DEFERRED）
- 一旦在同一安装中将 driver 加入选定仓库集合，仅扩展 `GITHUB_REPOS` 配置即可，无需修改 Deploy Approval 源码
- 未知的 / 第三方的仓库必须 fail closed
- 缺失、重复的仓库必须 fail closed。

## Review Agent 集成

Deploy Approval **不调用** Review Agent HTTP API，全部通过 GitHub PR Conversation 完成。
Deploy Approval **不需要**：
- Review Agent 主机 / IP / 端口
- Review Agent `GITHUB_TOKEN`
- Review Agent SSH / Dashboard
- `host.docker.internal` 网络连通性

Deploy Approval 仅通过 **GitHub PR 对话评论** 读取 Review Agent 输出。

### 评论协议

Review Agent 使用固定标记 `<!-- pr-review-agent -->` 写入评论，包含三个部分：

1. **Build Result** — `## PR Review Agent — Build Result`
   - 包含提交短 SHA、目标构建表和镜像引用。
2. **Test Results** — `## PR Review Agent — Test Results`（可选）
   - 包含测试套件通过/失败计数。
3. **Code Review** — `## PR Review Agent — Code Review`
   - 包含审查文本。

Deploy Approval 通过 `review_comment_parser.py` 解析这些评论并提取
`ReviewCommentEvidence`，包含 build_comment_id、builds、test/code review 来源。

### 可信评论作者

Deploy Approval 验证 Review Agent 评论来自 `secrets.yaml` 中配置的可信作者：

```yaml
review_comment_trust:
  author_id: "<review-agent-github-user-id>"
  author_login: "<review-agent-github-login>"
```

- `author_id` **必填**，必须是正整数。
- `author_login` 可选，但如果配置必须与 `comment.user.login` 匹配。
- `performed_via_github_app` 可能为 `null`（Review Agent 使用用户 PAT）。
- 来自其他作者的评论将被**拒绝（fail closed）**。

### 生产 Review Agent 单例不变量

**对于生产仓库（`4paradigm/phanthymotus`、`4paradigm/phanthymotus-driver`）：**

同一时刻最多只能存在一个权威的 Review Agent poller/producer。

原因：
- Review Agent poller watermark / processed comment IDs 是实例本地状态。
- 两台 Review Agent 服务器看到同一条 `/request_bot_review` 会产生重复的 job/comment/build。
- 不存在跨实例分布式去重。

**测试 Review Agent** 必须使用：
- 绝不允许监听生产仓库。

`AUTHORITATIVE_REVIEW_AGENT_COUNT_FOR_PRODUCTION_REPOS=1` 是上线人工门禁。

## 三层模型

1. GitHub PR `lifecycle comment hidden JSON`
2. Deploy Controller
3. 外部只读/执行系统：Review Agent、Agent Core、COS

GitHub hidden lifecycle JSON 是 Deploy Approval 唯一权威的持久化业务状态存储。Deploy Controller 不持久化业务状态，不依赖 SQLite、DB_PATH、DeploymentStore、local cursor、local lock、rollback state 或 webhook 双写。

Deploy Approval is restart-safe stateless, but intentionally single-replica / single-writer. Exactly one `GitHubCommandWatcher` serially processes mutating commands. Multiple concurrent Deploy Controller replicas are unsupported because the current hidden-state protocol has no CAS/distributed lock. Running replicas >1 would violate the at-most-once unsafe-side-effect model.

Polling reuses the upstream `POLL_INTERVAL_SECONDS` setting. Default: 30 seconds.
POLL_ENABLED must be true. Webhook is supplementary only.

## OPEN PR watcher enumeration

GitHubCommandWatcher 只枚举两个支持仓库中的所有 OPEN PR。

OPEN PR 枚举不使用 updated_at 年龄过滤，不使用 7-day lookback。

GitHub API 参数固定为：

- `state=open`
- `sort=updated`
- `direction=desc`
- `per_page=100`

从 `page=1` 开始持续翻页，直到 batch 为空或长度不足 100。没有 `page=5` / 500 PR 截断。

分页重叠按 PR number 去重。

不枚举 closed/merged PR。

如果枚举到的 PR 在命令执行前或 unsafe deploy POST 前发生 merge/close，现有 fresh PR gate 会拒绝它，ZERO deploy POST。

merge 后正式 release / production deployment 属于 main/release workflow，不属于 Deploy Approval。

Source matrix:

- GitHub PR comments: Review Agent output (build / test / code review evidence)
- Registry: only immutable verification / resolution of the exact `review_image_tag` from comments
- Agent Core: runtime identity / current `running_image` / MCP evidence
- GitHub hidden JSON: restart-safe persistence snapshot

`phanthymotus` 只部署 `perception` / `actucore`，`CORE` 不作为可部署组件；`phanthymotus-driver` 以 `driver_path` 作为机器策略身份，但 runtime id 必须通过 Agent Core 的精确 image repository 匹配得到，不能直接从 `driver_path` 拼接或模糊推导。Deploy Controller 通过本地管理员维护的 `machines.yaml` 中配置的 literal IPv4 `node_host` 连接到已存在的 Agent Core API，不做 Agent Core registration。Agent Core endpoint 固定为 `https://<node_host>:15678`，HTTP redirect 禁用。每台机器通过 Agent Core 的 `node_host` 直连 `https://<node_host>:15678`，不做 TLS peer certificate pin，允许 `verify=False`。Deploy Approval 不做 TOFU，不允许 PR/comment 指定目标 IP 或证书路径。Registry 只作为 `/request_deploy` 内部的 exact Review Agent image 解析与 immutable verification 实现细节，不作为独立 actor 或独立控制面。

fresh Review Agent build_results + fresh Registry immutable resolution
↓
fresh static component snapshot
↓
compare old/fresh snapshot
same snapshot:
preserve deployments
preserve runtime_id ONLY from old health-confirmed deployed component
changed snapshot:
replace static snapshot
clear deployment/case/COS validation state
NO rollback

Agent Core no-container response 只在 `running_image` / `error` 均缺失、`status` key 存在且 `logs` 为字符串时归一化为 `running_image=""`。`status` VALUE 不参与 CLEAN / health / case 决策；`error` 或 malformed shape 继续 fail closed。

unsafe deploy POST 进入未知结果时：
command.phase=uncertain
status=deploy-requested
ZERO later POST
NEW approve only

## Full-Coverage Machine Gate

/approve_deploy 必须首先通过 full-coverage gate：

1. 计算 ALL REMAINING component_ids
2. 计算 selected machine 实际能覆盖的 component_ids
3. 只有 machine_compatible_component_ids >= all_remaining_component_ids 才允许继续
4. 如果 coverage 不足：ZERO deploy POST，status 保持 deploy-requested，comment 列出 full-coverage machines

full-coverage gate 确保不再存在 partial machine-group 部署、不再跨机器轮询、不再等待剩余 component。

full-coverage gate 必须在 CLEAN gate 之前执行。

CLEAN 通过后：FULL-COVERAGE -> running_image-only CLEAN -> fresh exact approval comment -> final fresh PR/full HEAD -> persist command.phase=executing to GitHub FIRST -> deploy ALL REMAINING components -> durable status: testing -> fixed Case (advisory only) -> Machine Owner /record_test -> succeeded | failed.

不再检查 "所有 machine group 是否全部部署完毕"，因为 selected machine 已要求覆盖全部 REMAINING components。已 durably 成功的相同 snapshot 组件不会重新部署。

**无 post-deploy health gate：** 每次 Agent Core deploy POST 返回正常即视为该组件部署成功，不额外调用 `driver_status` 轮询 `running_image` 来判定生命周期成功。终态证据上传前，对已部署 runtime 做一次性的 `driver_status` 日志快照；失败只写固定 marker，不改变已写入 GitHub 的终态。

## Agent Authentication

## GitHub Installation Model

`GITHUB_INSTALLATION_ID` identifies a GitHub App installation on an
account/organization; it is **not** inherently one ID per repository.
A single installation can be configured for multiple selected repositories.

Current production auth intentionally uses one installation ID.
Current runtime validation target is `4paradigm/phanthymotus`.
`4paradigm/phanthymotus-driver` runtime authorization is **DEFERRED**.

When driver is enabled, prefer adding driver to the same installation's
selected repository set. Only introduce repo-to-installation routing
if GitHub later proves there are distinct installations.



- **Deploy Approval** uses GitHub App (`GITHUB_APP_ID`, `GITHUB_INSTALLATION_ID`, `GITHUB_APP_PRIVATE_KEY_FILE`).
- **Review Agent** uses a user-provided `GITHUB_TOKEN`. This is strictly separate from Deploy Approval's GitHub App credentials.

## Actor

时序图和合同只显示以下 Actor：

1. Developer
2. GitHub PR
3. Review Agent
4. Deploy Controller
5. Machine Owner
6. Agent Core
7. COS

不显示 GitHub State Proxy 和 Registry 作为独立 Actor。它们只能作为内部实现细节，不能成为第二个 Deploy Controller。

## 顶层状态

唯一合法顶层状态只有 7 个：

1. `review-required`
2. `reviewing`
3. `deploy-ready`
4. `deploy-requested`
5. `testing`
6. `succeeded`
7. `failed`

旧版的审批等待中间态、独立部署进行态以及拆分的部署/测试失败态，均不再作为实际顶层状态使用。

GitHub PR 的 `status:*` label 只是 projection，不能作为权威状态。允许的 label 精确为：

- `status: review-required`
- `status: reviewing`
- `status: deploy-ready`
- `status: deploy-requested`
- `status: testing`
- `status: succeeded`
- `status: failed`

label 更新顺序必须是：

1. 先写 hidden JSON
2. 再删除旧的 `status:*` label
3. 保留所有非 `status:*` label
4. 最后添加唯一新的 `status:<hidden-status>`

## Hidden JSON

hidden JSON 是唯一权威业务状态。至少包含：

- `version`
- `head_sha`
- `status`
- `review_evidence`
- `components`
- `deployments`
- `case_results`
- `test_result`
- `cos`
- `command`
- `last_processed_comment_id`

其中：

- `state.status` 是 authoritative lifecycle state
- `command.phase` 只允许 `completed`、`executing`、`uncertain`
- `uncertain` 不是 top-level lifecycle status；在 hidden state 中只允许作为 `command.phase=uncertain` 或 `approve_attempt.outcome=uncertain` 出现。

## 无状态边界

Deploy Controller 命令之间完全无状态。active runtime path 禁止依赖：

- SQLite
- DB_PATH
- DeploymentStore
- processed_comments table
- local CAS database
- local cursor database/file
- rollback state
- local machine lock database
- 旧版部署生命周期中间态及拆分失败态

旧文件可以暂时保留为兼容 stub，但 active server/runtime path 不得 import、实例化或调用这些旧持久化路径。

## command actor

`/request_deploy` 的 actor 必须是当前 PR Author。

`/approve_deploy machine=<alias>` 的 actor 必须满足以下任一条件：

- 是选中机器 `owners[]` 中的 owner
- repo permission 为 `write`、`maintain` 或 `admin`

`/record_test result=pass|fail [summary="..."]` 的 actor 必须满足以下任一条件：

- 是已实际部署机器的 owner
- repo permission 为 `write`、`maintain` 或 `admin`

## /request_deploy

`/request_deploy`：

- 零参数
- 只能由 PR Author 发起
- 先 fresh GET PR
- 再 fresh full HEAD
- 再 fresh GET PR 评论
- 通过 `review_comment_parser.extract_review_evidence()` 解析评论
- 验证评论作者为可信作者
- 验证 comment 中的 commit prefix resolve 到 fresh full HEAD
- 选择 latest unambiguous 同 HEAD review evidence
- 绑定 `review_evidence`（build_comment_id、test_comment_id、code_review_comment_id）
- 只取该 Job 中所有 successful deployable components
- CORE 排除
- `review_image_tag` 必须直接来自 Review Agent Build Result 评论的 Images section
- mutable image tag 只允许通过现有 Registry client 一次性解析成 immutable `repository@sha256:...`
- 保存 `resolved_platform`
- hidden JSON 持久化 validation snapshot
- `status: deploy-requested`

禁止重新引入这些参数：

- `build=`
- `image=`
- `target=`
- `test-mode=`
- `test-plan=`
- `test-case=`

## /approve_deploy：running_image-only CLEAN GATE

`/approve_deploy machine=<alias>` 每一条 NEW command 都必须重新读取：

- PR state
- full HEAD
- hidden lifecycle state
- command comment actor

若 HEAD drift：

- `current HEAD != hidden head_sha`
- ZERO deploy
- invalidate current validation
- `status: review-required`
- 下一步给 Developer：`/request_bot_review`

### Full-Coverage Machine Gate

一条 `/approve_deploy` 命令只选择一台 machine。这台 machine 必须覆盖 ALL REMAINING components。

- coverage 不完整 → ZERO deploy POST，status 保持 `deploy-requested`，提示用户选择 full-coverage machine
- 不存在跨 machine partial success
- 不存在"先部署一部分，再换另一台机器继续部署"的流程
- 成功覆盖并部署全部 REMAINING components 后直接进入 `testing`

### CLEAN GATE

CLEAN GATE 只读取 `running_image`，不判断机器状态。禁止把下面这些值用于 pre-deploy gate：

- `READY`
- `BUSY`
- `OFFLINE`
- `stopped`
- `running`
- 任何基于 `status` 字段与 `stopped` / `running` 组合出的 clean 条件
- node availability state
- machine readiness state

如果 selected machine 上的 ALL REMAINING components 都满足：

```text
running_image == ""
```

才允许继续。

如果任一 component 满足：

```text
running_image != ""
```

则：

- ZERO deploy POST
- `status` 继续为 `deploy-requested`
- 该 NEW approve command 自身完成
- cursor 推进到当前 comment id
- visible comment 明确提示：
  - 哪个 runtime/component 被占用
  - 当前 `running_image`
  - ZERO deployment was performed
  - Machine Owner 必须手工清空
  - 清空后再发一条 NEW `/approve_deploy machine=<alias>`

### 同一 machine 的多组件预检

本次 approval 的 ALL REMAINING components 必须在 ANY deploy POST 前全部 preflight：

```text
ALL REMAINING components preflight
BEFORE
ANY deploy POST
```

只要其中一个占用，必须 ZERO deploy POST，不能先部署一部分再发现另一个占用。

### CLEAN PASS 的严格顺序

只有全部 `running_image == ""` 时，才执行：

1. fresh exact approval comment revalidation
2. final fresh PR/full HEAD
3. persist `command.phase = executing` to GitHub FIRST
4. 然后才允许第一个 Agent Core deploy POST

严格顺序：

```text
CLEAN GATE PASS
    ↓
fresh exact approval comment revalidation (comment valid, id exact, actor exact, body parses, alias exact)
    ↓
any failure -> approve_attempt.outcome=approval_revoked, status=deploy-requested, command.phase=completed, ZERO POST
    ↓
final fresh PR/full HEAD (drift -> review-required, ZERO POST)
    ↓
GitHub hidden state command.phase=executing persisted
    ↓
POST existing Agent Core deploy
```

### fresh exact approval comment revalidation

任何检查失败：

- comment object valid
- comment id exact
- actor id exact
- body parses as approve_deploy
- machine alias exact

则：

- `approve_attempt.outcome=approval_revoked`
- `status=deploy-requested`
- `command.phase=completed`
- cursor advances to current comment
- ZERO deploy POST
- Machine Owner must send a NEW `/approve_deploy`

`approval_revoked` 不是 top-level status，不增加新的 lifecycle state。

## Case：advisory only

固定 case 只在所有 required components 都部署完之后、且 durable testing state 已写入 GitHub 之后运行，而且只作为 advisory evidence：

- 不得在所有 required components deploy 完成前运行
- 不得在 durable testing hidden state 写入前运行
- 不得在 testing label projection 前运行
- Case PASS 不得自动把状态改成 `succeeded`
- Case FAIL 不得阻止 Machine Owner 最终 `/record_test result=pass`
- Case FAIL 不得将 `testing` 改为 `failed`
- Case exception/timeout/unavailable 不得回退 `testing` 状态
- Case 必须使用实际 Agent Core binding / actual runtime id
- 不允许 placeholder PASS
- 不允许 shell / subprocess / SSH / user-supplied executable
- advisory case_results 持久化前必须 fresh-read hidden state 并校验 same HEAD + status=testing + same command

### 需要存在的真实行为测试

- `test_case_fail_does_not_block_overall_manual_pass`
- `test_case_not_run_before_all_components_deployed`
- `test_case_pass_does_not_auto_succeed`

## /record_test

`/record_test`：

- 只接受 `result=pass|fail [summary="..."]`
- 不接受 `machine=`
- 不接受 `evidence=`
- 不接受旧 `dpl_x` / `build=` / `test-mode=` 语法
- 先写 terminal GitHub state
- 之后才进行 COS upload（best effort）
- COS 失败不能回滚 terminal GitHub state

`/record_test` 的 GitHub 持久化顺序是：

1. `command.phase = completed`
2. `command.comment_id = current comment id`
3. `last_processed_comment_id = current comment id`
4. `result=pass` 时写 `test_result=pass` 和 `status=succeeded`
5. `result=fail` 时写 `test_result=fail` 和 `status=failed`
6. 先完成以上 terminal GitHub state，再上传 COS
7. COS 成功时只回填 `object_key`、`sha256`、`size` 到同一 terminal state

映射关系：

- `result=pass` -> `status: succeeded`
- `result=fail` -> `status: failed`

COS 默认归档只包含一个文件：

- `evidence.log.gz`

COS object key 固定为：

`phanthymotus_pr/<repo-dir>/<YYYY-MM>/<YYYY-MM-DD>/pr-<N>/evidence-<FULL_HEAD_SHA>.log.gz`

COS hidden state 只保存：

- `object_key`
- `sha256`
- `size`

GitHub 中只持久化 `object_key`、`sha256`、`size`。

终态 lifecycle comment 在 COS 上传成功后显示稳定的 Deploy Approval 下载链接：
GitHub PR comment
    ↓ [Download COS evidence]
120 秒 HTTPS COS presigned GET URL
    ↓
private COS object
    ↓
evidence.log.gz

COS presigned URL 由 Deploy Approval 在终端 COS 元数据重新绑定成功后生成，120 秒（2 分钟）短期有效。
该 URL 为临时持有者访问链接，点击后浏览器直接请求 COS，不经过 Deploy Approval。
COS 存储桶保持私有，不对外暴露公共 HTTPS 端点。
无需 GitHub 用户 OAuth、无需 OAuth 回调、无需 PKCE、无需每次点击的 Deploy Approval 授权网关。
不生成 /evidence/download 公共下载网关。
presigned URL 不会持久化写入 hidden state。
证据上传前已完成敏感信息脱敏、严格 UTF-8 处理、gzip 压缩、mtime=0、大小上限 10 MiB。


## restart / uncertain

restart 的固定合同：

```text
restart
    ↓
read hidden state
    ↓
command.phase == executing
    ↓
同一次 GitHub hidden-state write 中：
command.phase = uncertain
last_processed_comment_id = max(old last_processed_comment_id, command.comment_id)
    ↓
ZERO automatic replay
```

必须保证：

```text
last_processed_comment_id >= command.comment_id
```

watcher 不能缓存旧 cursor 再 reconcile。正确做法是：

```text
reconcile_pr()
    ↓
fresh read hidden state
    ↓
cursor = fresh_state.last_processed_comment_id
    ↓
fetch/filter comments
```

旧 `command.comment_id` 必须被消费，旧 comment 永远不能 automatic dispatch。

### uncertain 后必须重新 Review Evidence 验证

`executing -> uncertain` 之后，下一次 NEW `/approve_deploy` 需要：

1. fresh GET PR
2. fresh current full HEAD
3. fresh PR comments → `extract_review_evidence()`
4. Controller 本地 exact 过滤：可信作者 + commit prefix resolve + HEAD 绑定

如果：

- HEAD drift
- 或者 review evidence 缺失/不匹配

则：

- invalidate validation
- `status: review-required`
- 下一步给 Developer：`/request_bot_review`
- ZERO replay

如果 same HEAD + review evidence FOUND：

- refresh validation snapshot
- `status: deploy-requested`
- 然后才继续 running_image-only CLEAN GATE

## 最终命令流

### Developer

- `/request_bot_review`
- `/request_deploy`

### Machine Owner

- `/approve_deploy machine=<alias>`
- `/record_test result=pass|fail [summary="..."]`

### Read-only

- `/deploy_status`
- `/deploy_help [topic]`

## Required GitHub App Permissions

Minimum repository permissions:

- **Pull requests: write** — used for PR reads and PR conversation comments /
  status-label operations.
- **Metadata: read** — needed for collaborator permission lookup.

Do **not** request Administration or Actions permission.

**Contents: read** — The current target repository
`4paradigm/phanthymotus` is PUBLIC, so `resolve_commit_sha()` (which calls
`GET /repos/{repo}/commits/{ref}`) operates without granting Contents: read.
The current minimum deployment permissions therefore do not need to be
broadened solely for this public rollout. If a future authorized target
repository is private, the commit endpoint requires GitHub App `Contents:
read`; permissions must be explicitly reviewed before private-repo
enablement.

## 结论

Deploy Approval 的最终收口原则是：

- GitHub hidden JSON 是权威状态
- Deploy Controller 无状态
- Review Agent 不改接口
- Agent Core 不改 deploy contract
- CLEAN GATE 只看 `running_image`
- uncertain 后不自动 replay
- Case 只做 advisory

## 私钥操作合同

- 私钥绝不允许以明文形式出现在源码或文档中。
- 运行时通过 `GITHUB_APP_PRIVATE_KEY_FILE` 指定密钥文件路径。
- `deploy.sh` 强制校验本地私钥的属主/权限。
- 如果私钥已暴露，必须在运行时验证前轮换密钥。
- 绝不允许打印私钥内容。

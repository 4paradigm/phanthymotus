# Jev 超时回退默认模式

> 历史计划：其中保留本地精确回声拦截的条款已由 [移除本地回声拦截计划](remove-local-echo-filter.md) 取代。以下测试及部署记录保留原时间点含义，不作为移除过滤后的效果证据。

## 范围

- 用户要求语音和文字的 Jev 超时均回退当前默认 steer / interrupt / followup，不再因超时 reject。
- 保留共享截止时间，涵盖排队、准备、请求和重判；不使用迟到结果或旧的模型模式。消费时读取既有默认模式。
- 本地精确回声拒绝优先于超时回退；模型有效拒绝、非超时错误、身份不可读、队列满、停止与配置失效继续沿用原策略。
- 超时意味着未完成对象判断：旁人聊天、未匹配的回声可能进入默认处理。这是可用性取舍，不是模型确认接入。
- 仅修改 Core、测试与行为文档；不更改线上配置，不提交、推送或部署。

## 实施与验证

1. 调整 `_judge` 超时分支与排队过期处理，保留生命周期和至多一次交付。
2. 覆盖三种实时默认模式、请求/准备/排队/重判超时、迟到结果、精确回声、停止和失效；保留其他失败与拒绝回归。
3. 运行定向与 Core 全量测试，同步操作文档及原方案的现行超时说明。README 入口及外部 API 不变，无需改动。

## 验证状态

- 定向：`/tmp/jev-core-routing-test-20260921/bin/python -m pytest agent-core/tests/test_semantic_routing.py -q`：79 passed、2 subtests passed（3.73 秒）。
- Core 全量：同一解释器，临时独立 DB_PATH，`-m pytest agent-core/tests -q`：1389 passed、1 skipped、10 subtests passed（38.87 秒）。
- `git diff --check` 通过。已复核原计划与操作文档的超时说明；README 入口与外部请求契约未变。
- 此处为超时改动单独验证记录；随后用户授权与单次 Choice 改造一起部署，最新部署状态见 [单次判断计划](jev-single-choice-routing.md)。未提交、未真机验收。

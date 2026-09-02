"""llm_bench —— agent-core 的 LLM 入口回测工具。

按 `python3 tools/llm_bench` 的方式运行（见 __main__.py）。各模块之间用扁平的
`import config` 而非包相对导入，这样从目录直接运行和被 pytest 单独 import 都能work。
"""

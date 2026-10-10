"""
tests/test_asr_find_library_fork.py — 别在 import phonemizer 的时候 fork。

**`ctypes.util.find_library` 会 fork。** Linux 上它 `subprocess` 出一个
`ldconfig -p`，而 `phonemizer` 导入的 `dlinfo` 在 **import 时**就调它
（`dlinfo/_glibc.py` → `find_library('dl')`）。从一个 2.4 GB、几十个线程、其中一个
正在跑 TensorRT warmup 的进程里 fork，在 Orin 5 上把 ASR worker 永久卡死了：子进程
没走到 exec（fork 那一刻继承了别的线程持有的锁），父进程就一直等在
`_execute_child` 的错误管道上。import 锁不放，worker 再也没转写过一句，perception
整个 MCP 都不应答 —— 而且任何日志里都没有线索，因为最后写出来的那行正是 fork 前
的那行。

这是竞态：同一张卡在没别的东西加载时启动得很干净，所以它只在「启动控制」之后出现
（TTS warmup 和 ASR 卡片一起起来）。所以守的不是「更少发生」，而是**根本没有 fork**：

* 已知的几个库按路径解析，不进 subprocess；
* 不认识的库**照旧走原实现**，而不是返回 None —— 那会被调用方读成「没装」；
* 退出后必须还原，不能把替换留在 `ctypes.util` 上；
* 两个 worker 线程同时进来不能嵌套，否则内层还原时把外层的替换也抹掉。

Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import ctypes.util
import sys
import threading

import pytest

from vision_stubs import PERCEPTION_ROOT  # noqa: F401  (ROS stubs + sys.path)

import plugins.asr as asr_module  # noqa: E402


def test_known_library_resolves_without_subprocess(monkeypatch):
    """命中候选路径时，原实现一次都不该被调到。"""
    calls = []
    monkeypatch.setattr(ctypes.util, "find_library",
                        lambda name: calls.append(name) or "forked")
    monkeypatch.setattr(asr_module, "_FORK_FREE_LIBRARIES",
                        {"dl": ("/dev/null",)})
    with asr_module._find_library_without_forking():
        assert ctypes.util.find_library("dl") == "/dev/null"
    assert calls == [], f"仍然 fork 了：{calls}"


def test_missing_candidate_falls_back_to_the_real_lookup(monkeypatch):
    """路径都不存在时宁可 fork，也不能返回 None 让调用方以为库没装。"""
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: f"real:{name}")
    monkeypatch.setattr(asr_module, "_FORK_FREE_LIBRARIES",
                        {"dl": ("/nonexistent/libdl.so.2",)})
    with asr_module._find_library_without_forking():
        assert ctypes.util.find_library("dl") == "real:dl"


def test_unknown_library_is_untouched(monkeypatch):
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: f"real:{name}")
    with asr_module._find_library_without_forking():
        assert ctypes.util.find_library("sqlite3") == "real:sqlite3"


def test_the_patch_is_restored(monkeypatch):
    sentinel = lambda name: "sentinel"          # noqa: E731
    monkeypatch.setattr(ctypes.util, "find_library", sentinel)
    with asr_module._find_library_without_forking():
        assert ctypes.util.find_library is not sentinel
    assert ctypes.util.find_library is sentinel


def test_the_patch_is_restored_after_an_exception(monkeypatch):
    sentinel = lambda name: "sentinel"          # noqa: E731
    monkeypatch.setattr(ctypes.util, "find_library", sentinel)
    try:
        with asr_module._find_library_without_forking():
            raise RuntimeError("espeak exploded mid-import")
    except RuntimeError:
        pass
    assert ctypes.util.find_library is sentinel, "import 失败不能把替换留在那"


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="只有 Linux 的 find_library 会 fork ldconfig；"
                           "mac 上走 dyld，没有这个问题，候选表也用不到")
def test_dl_is_covered_on_linux():
    """`dl` 是真正要紧的那个 —— 在 Linux 上候选表必须命中，否则等于没修。

    aarch64 和 x86_64 的 glibc 路径都在表里（镜像是 aarch64，VPC 构建机是
    x86_64）。这条断言的意义是：表漏了会在**跑得到的机器上**失败，而不是在真机
    上静默退回 fork。
    """
    import os
    candidates = asr_module._FORK_FREE_LIBRARIES["dl"]
    assert any(os.path.exists(path) for path in candidates), \
        f"本机找不到 libdl，候选表要补：{candidates}"


def test_two_threads_do_not_nest_the_patch():
    """两个 worker 线程同时音素化时，不能有线程在外层还活着时被还原。"""
    original = ctypes.util.find_library
    seen_inside = []
    barrier = threading.Barrier(2, timeout=5)

    def body():
        with asr_module._find_library_without_forking():
            barrier.wait()
            seen_inside.append(ctypes.util.find_library is not original)

    threads = [threading.Thread(target=body) for _ in range(2)]
    for t in threads:
        t.start()
    # 两个线程都在块内时，各自看到的都必须是替换版 —— 嵌套还原会让其中一个看到原版
    for t in threads:
        t.join(timeout=5)
        assert not t.is_alive()
    assert seen_inside == [True, True], seen_inside
    assert ctypes.util.find_library is original

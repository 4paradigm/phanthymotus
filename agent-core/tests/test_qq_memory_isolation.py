"""Minimal QQ multi-chat memory and concurrency isolation; no SDK or LLM calls."""
import asyncio
import json
import pathlib
import sys
import time
from unittest import mock

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import collector
import config
import event
import prompt
from channel.manager import ChannelManager, qq_memory_scope
from channel import manager as channel_manager_module
from channel.tool_policy import allowed_for_principals
from event.llm import Event

APP = "1906322115"
CHAN = "qq_official_" + APP
PERSON_A = "ABCDEFGHIJKLMN"
PERSON_B = "PQRSTUVWXYZABCDE"
GROUP_1 = "GROUP111111111"
GROUP_2 = "GROUP222222222"


@pytest.fixture
def scope_store(monkeypatch):
    # Avoid reading/writing any live user database.
    state = {"channel_chat_memory": {}, "qq_c2c_owner_grant": {}}
    monkeypatch.setattr(config, "main", state)
    return state


def test_user_group_memory_isolated_and_can_forget(scope_store):
    mgr = ChannelManager()
    locations = [(CHAN, PERSON_A), (CHAN, PERSON_B),
                 (CHAN, GROUP_1), (CHAN, GROUP_2),
                 ("another_qq_bot", PERSON_A)]
    for n, loc in enumerate(locations):
        assert mgr.remember_chat_memory(*loc, f"PRIVATE_FACT_{n}")
    for n, loc in enumerate(locations):
        assert mgr.get_chat_memory(*loc) == [f"PRIVATE_FACT_{n}"]
    assert mgr.forget_chat_memory(CHAN, PERSON_A, "PRIVATE_FACT_0") == 1
    assert mgr.get_chat_memory(CHAN, PERSON_A) == []
    assert mgr.get_chat_memory(CHAN, PERSON_B) == ["PRIVATE_FACT_1"]
    assert mgr.get_chat_memory(CHAN, GROUP_1) == ["PRIVATE_FACT_2"]
    assert mgr.forget_chat_memory(CHAN, GROUP_1) == 1
    assert mgr.get_chat_memory(CHAN, GROUP_2) == ["PRIVATE_FACT_3"]


def test_memory_dedup_capacity_and_validation(scope_store):
    mgr = ChannelManager()
    assert mgr.remember_chat_memory(CHAN, PERSON_A, "a fact")
    assert not mgr.remember_chat_memory(CHAN, PERSON_A, "a fact")
    for i in range(30):
        mgr.remember_chat_memory(CHAN, PERSON_A, f"memory-{i}")
    entries = mgr.get_chat_memory(CHAN, PERSON_A)
    assert len(entries) == 20 and entries[-1] == "memory-29"
    assert "a fact" not in entries
    with pytest.raises(ValueError):
        mgr.remember_chat_memory(CHAN, PERSON_A, " " * 10)
    with pytest.raises(ValueError):
        mgr.remember_chat_memory(CHAN, PERSON_A, "x" * 501)


def test_qq_prompt_excludes_global_agent_memory(scope_store, monkeypatch, tmp_path):
    system = tmp_path / "system.txt"
    global_memory = tmp_path / "private.txt"
    system.write_text("PUBLIC_AGENT_INSTRUCTIONS", encoding="utf-8")
    global_memory.write_text("GLOBAL_OWNER_PRIVATE_DO_NOT_DISCLOSE", encoding="utf-8")
    scope_store["event"] = {"llm": {
        "prompt_system": str(system), "prompt_memory": str(global_memory),
    }}
    monkeypatch.setattr(prompt, "_env_static", lambda *args: "")
    monkeypatch.setattr(prompt, "_l1_cache", {"mtime": 0, "content": ""})
    mgr = channel_manager_module.manager
    mgr.remember_chat_memory(CHAN, PERSON_A, "ONLY_A_CAN_READ")
    mgr.remember_chat_memory(CHAN, GROUP_1, "ONLY_GROUP1_CAN_READ")
    a = prompt.build_system({}, qq_turn=True, qq_scope=(CHAN, PERSON_A))["content"]
    group = prompt.build_system({}, qq_turn=True, qq_scope=(CHAN, GROUP_1))["content"]
    local = prompt.build_system({})["content"]
    assert a == group  # QQ system prefixes must be stable across chats.
    assert "ONLY_A_CAN_READ" not in a and "ONLY_GROUP1_CAN_READ" not in group
    def scoped_event(scope):
        return {"source":"collector", "text":"QQ event", "_qq_channel_turn":True,
                "_qq_memory_facts":mgr.get_chat_memory(*scope)}
    user_a = prompt.build({"role":"system", "content":a}, [], scoped_event((CHAN, PERSON_A)))[-1]["content"]
    user_g = prompt.build({"role":"system", "content":group}, [], scoped_event((CHAN, GROUP_1)))[-1]["content"]
    assert "ONLY_A_CAN_READ" in user_a and "ONLY_A_CAN_READ" not in user_g
    assert "ONLY_GROUP1_CAN_READ" in user_g and "ONLY_GROUP1_CAN_READ" not in user_a
    mgr.remember_chat_memory(CHAN, PERSON_A, "NEW_A_FACT")
    assert prompt.build_system({}, qq_turn=True, qq_scope=(CHAN, PERSON_A))["content"] == a
    assert "GLOBAL_OWNER_PRIVATE_DO_NOT_DISCLOSE" not in a
    assert "GLOBAL_OWNER_PRIVATE_DO_NOT_DISCLOSE" not in group
    assert "GLOBAL_OWNER_PRIVATE_DO_NOT_DISCLOSE" in local


def test_qq_prompt_hides_global_task_status_and_shared_skills(monkeypatch):
    import task_store
    monkeypatch.setattr(task_store, "active_tasks",
                        lambda: (_ for _ in ()).throw(AssertionError("Global tasks leaked")))
    assert "<status time=" in prompt._env_dynamic(isolated=True)
    with mock.patch.object(sys.modules["event.skills"], "visible_skills", return_value=[{
        "slug": "private", "name": "Private", "oneLiner": "PRIVATE_SKILL_INFO"
    }]):
        monkeypatch.setattr(prompt, "_l2_static_cache",
                            {"fingerprint": None, "content": ""})
        safe = prompt._env_static({}, set(), False)
        assert "PRIVATE_SKILL_INFO" not in safe


def test_same_token_group_and_private_have_different_keys():
    dm = _qq_event(PERSON_A, "private", group=False)
    group = _qq_event(PERSON_A, "group", group=True)
    batches = collector._split_by_bot_trust([dm, group])
    assert len(batches) == 2
    first = collector._build_trigger(batches[0], True)["_qq_chat_id"]
    second = collector._build_trigger(batches[1], True)["_qq_chat_id"]
    assert first == "p2p:" + PERSON_A
    assert second == "group:" + PERSON_A


def test_memory_tools_use_only_current_turn_scope(scope_store):
    async def go():
        token = qq_memory_scope.set((CHAN, PERSON_A))
        try:
            answer = await event.memory.channel_memory_remember("alpha private preference")
            assert "已记入" in answer
            assert "alpha private preference" in await event.memory.channel_memory_read()
        finally:
            qq_memory_scope.reset(token)
        token = qq_memory_scope.set((CHAN, GROUP_1))
        try:
            assert "alpha private preference" not in await event.memory.channel_memory_read()
            await event.memory.channel_memory_remember("group fact")
            await event.memory.channel_memory_forget()
        finally:
            qq_memory_scope.reset(token)
        assert "Error:" in await event.memory.channel_memory_read()
        assert "Error:" in await event.memory.channel_memory_remember("forbidden")
    asyncio.run(go())
    mgr = ChannelManager()
    assert mgr.get_chat_memory(CHAN, PERSON_A) == ["alpha private preference"]
    assert mgr.get_chat_memory(CHAN, GROUP_1) == []


def test_qq_tools_cannot_call_global_memory_even_as_owner(scope_store, monkeypatch):
    from channel import acl
    from channel.tool_policy import _qq_safe
    owner_uid = f"official:{APP}:{PERSON_A}"
    scope_store["qq_c2c_owner_grant"] = {"enabled": True, "principal": owner_uid}
    monkeypatch.setattr(acl, "get_user", lambda platform, principal:
                        {"role": "owner", "tool_filter": "*"} if principal == owner_uid else None)
    assert allowed_for_principals([("qq", owner_uid)], "channel_memory_remember")
    for admin_tool in ("update_memory", "search_history", "memory_recall", "raw_input_info"):
        assert allowed_for_principals([("qq", owner_uid)], admin_tool), admin_tool
    assert not _qq_safe("channel_memory_remember", role="viewer", user_id="invalid")


def test_viewer_and_verified_group_can_use_only_scoped_memory(scope_store, monkeypatch):
    from channel import acl, qq_groups
    from channel.tool_policy import allowed_for_principals
    user = f"official:{APP}:{PERSON_B}"
    group_user = f"official:{APP}:group:{GROUP_1}:{PERSON_B}"
    monkeypatch.setattr(acl, "get_user", lambda *a: {
        "role": "viewer", "tool_filter": "mcp__channel__channel_reply,finish"})
    monkeypatch.setattr(qq_groups, "get_group", lambda ch, gid:
                        {"status": "approved"} if gid == GROUP_1 else {})
    monkeypatch.setattr(channel_manager_module, "_get_channel_configs",
                        lambda: [{"id": CHAN, "platform": "qq",
                                  "config": {"app_id": APP}}])
    assert allowed_for_principals([("qq", user)], "channel_memory_read")
    assert allowed_for_principals([("qq", user)], "channel_memory_remember")
    assert allowed_for_principals([("qq", group_user)], "channel_memory_forget")
    assert not allowed_for_principals([("qq", user)], "update_memory")
    assert not allowed_for_principals([("qq", user)], "Bash")


def _qq_event(chat_id, msg_id, *, group=False):
    identity = (f"official:{APP}:group:{chat_id}:{PERSON_A}" if group
                else f"official:{APP}:{chat_id}")
    return {"source": "dds:/channel/request/" + CHAN,
            "text": json.dumps({"platform": "qq", "channel_id": CHAN,
                                "chat_id": chat_id, "user_id": identity,
                                "sender_type": "user", "user_role": "viewer",
                                "message_id": msg_id, "text": "message",
                                "chat_type": "group" if group else "p2p"}),
            "ts": time.time()}


def test_interleaved_users_and_groups_separated_by_collector(scope_store):
    evs = [_qq_event(PERSON_A, "m1"), _qq_event(PERSON_B, "m2"),
           _qq_event(GROUP_1, "m3", group=True),
           _qq_event(GROUP_1, "m4", group=True),
           _qq_event(GROUP_2, "m5", group=True),
           _qq_event(PERSON_A, "m6")]
    batches = collector._split_by_bot_trust(evs)
    assert [len(b) for b in batches] == [1, 1, 2, 1, 1]
    ids = [collector._build_trigger(batch, True)["_qq_chat_id"] for batch in batches]
    assert ids == ['p2p:' + PERSON_A, 'p2p:' + PERSON_B,
                   'group:' + GROUP_1, 'group:' + GROUP_2, 'p2p:' + PERSON_A]
    assert all(collector._build_trigger(b, True)["_qq_channel_turn"] for b in batches)


def test_coroutines_do_not_share_contextvars_or_reply_scope(scope_store, monkeypatch):
    agent = Event()
    async def fake_llm(trigger, cancel=None, reconsider=None):
        current_scope = qq_memory_scope.get()
        assert current_scope == ((CHAN, trigger["_qq_chat_id"]) if trigger.get("_qq_channel_turn") else None)
        await asyncio.sleep(0)
        return qq_memory_scope.get()
    monkeypatch.setattr(agent, "_one_turn_impl", fake_llm)
    async def go():
        values = await asyncio.gather(
            *(agent._one_turn({
                "_qq_channel_turn": True, "_qq_channel_id": CHAN,
                "_qq_chat_id": cid, "_channel_principals": [("qq", "verified")],
            }) for cid in (PERSON_A, PERSON_B, GROUP_1, GROUP_2)))
        assert values == [(CHAN, c) for c in (PERSON_A, PERSON_B, GROUP_1, GROUP_2)]
        assert qq_memory_scope.get() is None
        await agent._one_turn({"_qq_channel_turn": False, "_qq_channel_id": CHAN,
                               "_qq_chat_id": PERSON_A})
        assert qq_memory_scope.get() is None
    asyncio.run(go())


def test_owner_skill_rebuild_keeps_private_prompt(scope_store, monkeypatch, tmp_path):
    """Owner activate/deactivate must preserve QQ isolation on subsequent request."""
    import inspect
    sysfile=tmp_path/"safe-system.txt"
    globfile=tmp_path/"private.txt"
    sysfile.write_text("PUBLIC_SYSTEM",encoding="utf-8")
    globfile.write_text("LOCAL_PRIVATE_IDENTITY",encoding="utf-8")
    scope_store["event"]={"llm":{"prompt_system":str(sysfile),"prompt_memory":str(globfile)}}
    monkeypatch.setattr(prompt,"_env_static",
                        lambda registry,bound,shared:
                        "GLOBAL_PRIVATE_SKILL" if shared else "SAFE_QQ_STATIC")
    scope=(CHAN,"p2p:"+PERSON_A)
    before=prompt.build_system({},set(),qq_turn=True,qq_scope=scope)["content"]
    after=prompt.build_system({},set(),qq_turn=True,qq_scope=scope)["content"]
    assert before==after and "SAFE_QQ_STATIC" in after
    assert "GLOBAL_PRIVATE_SKILL" not in after
    assert "LOCAL_PRIVATE_IDENTITY" not in after
    src=inspect.getsource(Event._one_turn_impl)
    target=src.split("if any(c['function']['name'] in skill_tools for c in tool_calls):",1)[1].split("\n\n",1)[0]
    assert "qq_turn=qq_private_turn" in target
    assert "qq_scope=qq_memory_scope.get() if qq_private_turn else None" in target


def test_qq_prompt_system_never_queries_scoped_sqlite(scope_store,monkeypatch,tmp_path):
    sysfile=tmp_path/"safe-system.txt"
    globfile=tmp_path/"local-memory.txt"
    sysfile.write_text("PUBLIC_SYSTEM",encoding="utf-8")
    globfile.write_text("LOCAL_PRIVATE_MEMORY",encoding="utf-8")
    scope_store["event"]={"llm":{"prompt_system":str(sysfile),"prompt_memory":str(globfile)}}
    monkeypatch.setattr(prompt,"_env_static",lambda *args:"")
    with mock.patch.object(channel_manager_module.manager,"get_chat_memory",
                           side_effect=AssertionError("sync sqlite read in system prompt")):
        system=prompt.build_system({},qq_turn=True,qq_scope=(CHAN,PERSON_A))["content"]
    assert "PUBLIC_SYSTEM" in system
    assert "LOCAL_PRIVATE_MEMORY" not in system


def test_scoped_facts_read_uses_background_worker(scope_store):
    def blocked(*_):
        time.sleep(.12)
        return ["ONLY_THIS_QQ_USER"]
    async def scenario():
        with mock.patch.object(channel_manager_module.manager,"get_chat_memory",side_effect=blocked):
            pending=asyncio.create_task(prompt.read_qq_scoped_facts((CHAN,PERSON_A)))
            await asyncio.wait_for(asyncio.sleep(.015),timeout=.075)
            assert not pending.done()
            assert await pending==["ONLY_THIS_QQ_USER"]
        assert await prompt.read_qq_scoped_facts(None)==[]
    asyncio.run(scenario())

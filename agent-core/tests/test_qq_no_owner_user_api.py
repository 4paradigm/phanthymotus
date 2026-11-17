"""QQ does not permit privileged roles through the shared Users endpoint."""
from unittest import mock
import fastapi
import pytest
from api import channel as channel_api

QQ_ID = "official:1900000000:UserOpenId123456789AB"


@pytest.mark.parametrize("role", ["owner", "operator"])
def test_generic_qq_user_api_rejects_privileged_roles(role):
    with mock.patch.object(channel_api.acl, "upsert_user") as save:
        with pytest.raises(fastapi.HTTPException) as caught:
            channel_api.upsert_user(channel_api.UpsertUserReq(
                platform="qq", user_id=QQ_ID, role=role, tool_filter="*"
            ))
    assert caught.value.status_code == 400
    save.assert_not_called()


def test_generic_qq_user_api_rejects_custom_tool_filter():
    with mock.patch.object(channel_api.acl, "upsert_user") as save:
        with pytest.raises(fastapi.HTTPException) as caught:
            channel_api.upsert_user(channel_api.UpsertUserReq(
                platform="qq", user_id=QQ_ID, role="viewer", tool_filter="Bash"
            ))
    assert caught.value.status_code == 400
    save.assert_not_called()


@pytest.mark.parametrize("role, expected", [
    ("viewer", "mcp__channel__channel_reply,finish"),
    ("blocked", ""),
])
def test_generic_qq_user_api_keeps_chat_and_block(role, expected):
    with mock.patch.object(channel_api.acl, "upsert_user",
                           return_value={"role": role}) as save:
        result = channel_api.upsert_user(channel_api.UpsertUserReq(
            platform="qq", user_id=QQ_ID, role=role
        ))
    assert result["user"]["role"] == role
    assert save.call_args.args[4] == expected


def test_generic_qq_user_api_normalizes_legacy_wildcard_to_chat():
    with mock.patch.object(channel_api.acl, "upsert_user",
                           return_value={"role": "viewer"}) as save:
        channel_api.upsert_user(channel_api.UpsertUserReq(
            platform="qq", user_id=QQ_ID, role="viewer", tool_filter="*"
        ))
    assert save.call_args.args[4] == "mcp__channel__channel_reply,finish"


def test_generic_user_api_preserves_feishu_owner_semantics():
    with mock.patch.object(channel_api.acl, "upsert_user",
                           return_value={"role": "owner"}) as save:
        result = channel_api.upsert_user(channel_api.UpsertUserReq(
            platform="feishu", user_id="ou_test", role="owner", tool_filter="*"
        ))
    assert result["user"]["role"] == "owner"
    assert save.call_args.args[4] == "*"

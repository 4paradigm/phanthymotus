from tool_config import config_error_message


def test_depth_config_reason_is_not_replaced_with_credentials():
    reason = 'DepthART files are not provisioned'
    assert config_error_message({'adapter_ok': False, 'message': reason}) == reason
    assert config_error_message({'status': 'error', 'detail': reason}) == reason


def test_config_success_and_generic_failure():
    assert config_error_message({'status': 'configured'}) == ''
    assert config_error_message({'status': 'loading', 'message': 'Downloading'}) == ''
    assert config_error_message({'adapter_ok': False}) == 'Configuration rejected by plugin'

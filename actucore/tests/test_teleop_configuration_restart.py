"""Configuration persistence must not restore movement or override a new site file."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'plugins'))
from teleop.plugin import TeleopPlugin

@pytest.fixture
def cfg(tmp_path):
    return {'mode':'shadow','position_scale':.5,
            'capture':{'state_file':str(tmp_path/'capture.json'),'secret':'not-a-setting'}}

def test_restart_restores_config_only(cfg):
    first=TeleopPlugin(cfg,None)
    assert first.dispatch('teleop',{'action':'config','mode':'live','position_scale':.65})['state']=='idle'
    saved=json.loads(first._config_file.read_text())
    assert 'capture' not in saved['values'] and 'secret' not in first._config_file.read_text()
    assert first._config_file.stat().st_mode & 0o777==0o600
    second=TeleopPlugin(cfg,None)
    assert second.info()['configuration']['position_scale']==.65
    assert second.info()['mode']=='live'
    assert second.runtime is None and second.link is None and not second.info()['output_active']

def test_rejected_config_does_not_replace_saved_value(cfg):
    card=TeleopPlugin(cfg,None)
    card.dispatch('teleop',{'action':'config','position_scale':.65})
    before=card._config_file.read_bytes()
    assert card.dispatch('teleop',{'action':'config','position_scale':2})['error']
    assert card._config_file.read_bytes()==before
    card.runtime=SimpleNamespace(status=lambda:{'authority_valid':True})
    assert card.dispatch('teleop',{'action':'config','position_scale':.7})['error']=='release_before_config'
    assert card._config_file.read_bytes()==before

def test_new_site_configuration_supersedes_old_cache(cfg):
    card=TeleopPlugin(cfg,None)
    card.dispatch('teleop',{'action':'config','mode':'live','position_scale':.65})
    updated=copy.deepcopy(cfg);updated['position_scale']=.4
    next_card=TeleopPlugin(updated,None)
    assert next_card.cfg['mode']=='shadow' and next_card.cfg['position_scale']==.4

def test_corrupt_config_blocks_host_until_explicit_configuration(cfg):
    card=TeleopPlugin(cfg,None);card._config_file.write_text('{broken')
    restarted=TeleopPlugin(cfg,None)
    assert restarted.dispatch('teleop',{'action':'info'})['error']=='saved_configuration_invalid'
    assert restarted.dispatch('teleop',{'action':'config','position_scale':.5})['state']=='idle'
    assert TeleopPlugin(cfg,None)._config_error is None

def test_failed_atomic_write_retains_old_config(cfg,monkeypatch):
    card=TeleopPlugin(cfg,None)
    card.dispatch('teleop',{'action':'config','position_scale':.65})
    before=card._config_file.read_bytes()
    def fail(*args):raise OSError('disk full')
    monkeypatch.setattr('teleop.plugin.os.replace',fail)
    assert card.dispatch('teleop',{'action':'config','position_scale':.7})['error']=='disk full'
    assert card.cfg['position_scale']==.65 and card._config_file.read_bytes()==before


@pytest.mark.parametrize('values,reason', [
    ({'robot_profile':'g1_23','shadow_feedback_source':'driver_joints','mode':'live'}, 'joints_feedback_requires_g1_shadow'),
    ({'robot_profile':'tianyi2','shadow_feedback_source':'driver_joints'}, 'joints_feedback_requires_g1_shadow'),
    ({'namespace':'robot-name'}, 'invalid_namespace'),
    ({'namespace':'robot/name'}, 'invalid_namespace'),
    ({'driver_mcp_url':'http://192.0.2.1:15707/mcp'}, 'driver_mcp_must_be_loopback'),
    ({'driver_mcp_url':'http://127.0.0.1:15707/mcp?secret=value'}, 'driver_mcp_must_be_loopback'),
    ({'driver_mcp_url':'http://127.0.0.1:99999/mcp'}, 'driver_mcp_must_be_loopback'),
])
def test_invalid_connection_settings_preserve_running_host_and_saved_config(cfg,values,reason):
    card=TeleopPlugin(cfg,None)
    card.dispatch('teleop',{'action':'config','position_scale':.65})
    before=card._config_file.read_bytes()
    host=SimpleNamespace(status=lambda:{'authority_valid':False})
    card.runtime=host;card.link=SimpleNamespace(lease=None)
    result=card.dispatch('teleop',{'action':'config',**values})
    assert result['error']==reason
    assert card._config_file.read_bytes()==before and card.runtime is host


def test_g1_shadow_configuration_and_action_capabilities(cfg):
    card=TeleopPlugin(cfg,None)
    result=card.dispatch('teleop',{'action':'config','robot_profile':'g1_23',
        'shadow_feedback_source':'driver_joints','namespace':'g1_bj',
        'driver_mcp_url':'http://127.0.0.1:15705/mcp'})
    assert result['state']=='idle'
    actions=card.get_tools()[0]['inputSchema']['properties']['action']['enum']
    assert {'start','stop','calibrate','open_pairing'}<=set(actions)
    assert not {'finish','record_start','record_stop','record_status'} & set(actions)
    assert card.dispatch('teleop',{'action':'finish'})['error']=='unsupported_action'
    assert card.runtime is None
    assert 'finish' in TeleopPlugin({},None).get_tools()[0]['inputSchema']['properties']['action']['enum']

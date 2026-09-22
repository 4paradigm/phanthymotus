"""Validate Canvas's explicit Driver binding before arming a teleop card."""
from .adapter import validate_driver_endpoint


def validate_binding(value, profile):
    fields={'mcp_id','tool','url','namespace','command_topic','feedback_topic','robot_profile','protocol_version'}
    if not isinstance(value,dict) or set(value)!=fields:raise ValueError('driver_binding_required')
    if not isinstance(value['mcp_id'],str) or not value['mcp_id']:raise ValueError('driver_binding_invalid')
    if type(value['protocol_version']) is not int or value['protocol_version']!=1:
        raise ValueError('driver_protocol_mismatch')
    if value['tool']!='teleop_executor' or value['robot_profile']!=profile:
        raise ValueError('driver_profile_mismatch')
    if not isinstance(value['namespace'],str):raise ValueError('invalid_namespace')
    validate_driver_endpoint({'namespace':value['namespace'],'driver_mcp_url':value['url']})
    prefix=f"/{value['namespace']}/motion/teleop"
    if value['command_topic']!=prefix+'/command' or value['feedback_topic']!=prefix+'/feedback':
        raise ValueError('driver_topic_mismatch')
    return dict(value)

"""Validate Canvas's explicit Driver binding before arming a teleop card."""
from .adapter import validate_driver_endpoint


def validate_binding(value, profile):
    fields={'mcp_id','tool','url','namespace','command_topic','feedback_topic','robot_profile','protocol_version'}
    if isinstance(value,dict) and value.get('protocol_version')==2:
        if type(value['protocol_version']) is not int or set(value)!=fields|{'execution_binding'}:
            raise ValueError('driver_binding_required')
        if value['tool']!='motion_control' or profile!='tianyi2' or value['robot_profile']!=profile:
            raise ValueError('driver_profile_mismatch')
        validate_driver_endpoint({'namespace':value['namespace'],'driver_mcp_url':value['url']})
        prefix=f"/{value['namespace']}/motion"
        if value['command_topic']!=prefix+'/control/command' or value['feedback_topic']!=prefix+'/teleop/feedback':
            raise ValueError('driver_topic_mismatch')
        execution=value['execution_binding']
        if not isinstance(execution,dict) or set(execution)!=fields|{'resources'}:
            raise ValueError('execution_binding_required')
        if (execution.get('tool')!='arm' or execution.get('resources')!=['arm_l','arm_r']
                or any(execution.get(k)!=value[k] for k in ('mcp_id','url','namespace','robot_profile','protocol_version','feedback_topic'))
                or execution.get('command_topic')!=prefix+'/arm/command'):
            raise ValueError('execution_binding_mismatch')
        if not isinstance(value['mcp_id'],str) or not value['mcp_id']:raise ValueError('driver_binding_invalid')
        return {**value,'execution_binding':dict(execution)}
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

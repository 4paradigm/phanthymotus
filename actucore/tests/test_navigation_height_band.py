"""Shared obstacle-band regression. Launch harness is ROS-free, not live bringup."""
from __future__ import annotations

import copy
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / 'plugins/controlled_semantic_spatial'
NAV2 = PLUGIN / 'runtime/nav2'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(NAV2))
sys.path.insert(0, str(Path(__file__).parent))

from nav2.height_band import apply_obstacle_height_band
from plugins.controlled_semantic_spatial.mapping.contract import FAST_LIVO2_CONFIG_DEFAULTS
from plugins.controlled_semantic_spatial.mapping.plugin import FastLivo2Plugin
from plugins.controlled_semantic_spatial.plugin import ControlledSemanticSpatialPlugin
from plugins.controlled_semantic_spatial.runtime import ControlledSemanticSpatialRuntime
import test_controlled_semantic_spatial_plugin as fixtures


def template():
    return yaml.safe_load((NAV2 / 'config/nav2_params.yaml').read_text())


def limits(data):
    result = []
    for name in ('local_costmap', 'global_costmap'):
        layer = data[name][name]['ros__parameters']['obstacle_layer']
        for target in (layer, layer['registered_cloud']):
            result.append((target['min_obstacle_height'], target['max_obstacle_height']))
    return result


class HeightBandConfigTest(unittest.TestCase):
    def test_template_defaults_match_mapping_and_preserve_inflation_radius(self):
        data = template()
        band = (FAST_LIVO2_CONFIG_DEFAULTS['obstacle_min_height_m'],
                FAST_LIVO2_CONFIG_DEFAULTS['obstacle_max_height_m'])
        self.assertEqual(band, (-.30, .30))
        self.assertEqual(limits(data), [band] * 4)
        for name in ('local_costmap', 'global_costmap'):
            params = data[name][name]['ros__parameters']
            self.assertEqual(params['inflation_layer']['inflation_radius'], .55)

    def test_overwrites_all_eight_leaves_including_legacy_missing_layer_keys(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                data = template()
                if legacy:
                    for name in ('local_costmap', 'global_costmap'):
                        layer = data[name][name]['ros__parameters']['obstacle_layer']
                        del layer['min_obstacle_height'], layer['max_obstacle_height']
                        layer['registered_cloud'].update(min_obstacle_height=-3., max_obstacle_height=3.)
                before = copy.deepcopy(data)
                apply_obstacle_height_band(data, -.5, .3)
                self.assertEqual(limits(data), [(-.5, .3)] * 4)
                for name in ('local_costmap', 'global_costmap'):
                    p = data[name][name]['ros__parameters']
                    self.assertEqual(p['inflation_layer'], before[name][name]['ros__parameters']['inflation_layer'])
                    self.assertEqual(p['footprint'], before[name][name]['ros__parameters']['footprint'])
                    self.assertEqual(p['obstacle_layer']['registered_cloud']['obstacle_min_range'], .2)

    def test_invalid_bands_do_not_mutate_configuration(self):
        for lo, hi in ((True,.3),(-.5,False),('-.5',.3),(float('nan'),.3),(-.5,float('inf')),(-3.1,.3),(-.5,3.1),(.3,.3),(.5,.3)):
            with self.subTest(lo=lo,hi=hi):
                data = template(); before = copy.deepcopy(data)
                with self.assertRaises(ValueError): apply_obstacle_height_band(data,lo,hi)
                self.assertEqual(data,before)

    def test_wrong_map_frame_layer_or_source_rejected_before_any_band_mutation(self):
        mutations = [
            lambda p:p.update(global_frame='odom'),
            lambda p:p['obstacle_layer'].update(plugin='different::Plugin'),
            lambda p:p['obstacle_layer'].update(observation_sources='other'),
            lambda p:p['obstacle_layer'].pop('registered_cloud'),
            lambda p:p.update(plugins=['inflation_layer']),
        ]
        for mutate in mutations:
            data=template(); mutate(data['global_costmap']['global_costmap']['ros__parameters']);before=copy.deepcopy(data)
            with self.assertRaises(ValueError):apply_obstacle_height_band(data,-.5,.3)
            self.assertEqual(data,before)


class HeightBandLifecycleTest(unittest.TestCase):
    def test_real_mapping_config_matches_adapter_command_and_both_runtime_launches(self):
        class Backend:
            def __init__(self):self.calls=[]
            def info(self):return {'state':'ready','backend':'fast_livo2_ros_topic','bridge_subscribers':1}
            def execute(self,action,args):
                self.calls.append((action,copy.deepcopy(args)))
                assert action=='configure_obstacle_filter'
                return {'status':'configured'}
            def stop(self):pass
        backend=Backend();mapping=FastLivo2Plugin({},None,backend=backend);runtime=fixtures.FakeRuntime()
        plugin=ControlledSemanticSpatialPlugin({},'ubuntu',None,runtime=runtime,mapping_plugin=mapping,
                 planning_plugin=fixtures.FakeComponent('planning'),semantic_plugin=fixtures.FakeComponent('semantic'))
        request={'action':'start','input_bindings':fixtures._external_bindings_for_start()}
        for band in ((-.5,.3),(-.8,.2)):
            result=plugin.dispatch(plugin.PREFIX,{'action':'config','obstacle_min_height_m':band[0],'obstacle_max_height_m':band[1]})
            self.assertEqual(result['state'],'configured')
            self.assertEqual(plugin.dispatch(plugin.PREFIX,request)['state'],'ready')
            self.assertEqual((runtime.start_kwargs['obstacle_min_height_m'],runtime.start_kwargs['obstacle_max_height_m']),band)
            self.assertEqual((backend.calls[-1][1]['min_height_m'],backend.calls[-1][1]['max_height_m']),band)
            kwargs=runtime.start_kwargs
            args=ControlledSemanticSpatialRuntime._launch_arguments(kwargs['namespace'],kwargs['input_topics'],
                kwargs['control_interface'],kwargs['rotate_reengage_rad'],kwargs['obstacle_min_height_m'],kwargs['obstacle_max_height_m'])
            for target in ('fast_livo2','nav2'):
                values=dict(s.split(':=',1) for s in args[target])
                self.assertEqual((float(values['obstacle_min_height_m']),float(values['obstacle_max_height_m'])),band)
            self.assertEqual(plugin.dispatch(plugin.PREFIX,{'action':'config','obstacle_min_height_m':-.2})['error_code'],'config_while_running')
            self.assertEqual(plugin.dispatch(plugin.PREFIX,{'action':'stop'})['state'],'idle')

    def test_missing_or_bad_mapping_band_never_starts_runtime(self):
        payloads=[{}, {'config':None}, {'config':{}}, {'config':{'obstacle_min_height_m':-.5}},
                  {'config':{'obstacle_min_height_m':float('nan'),'obstacle_max_height_m':.3}},
                  {'config':{'obstacle_min_height_m':.4,'obstacle_max_height_m':.3}},
                  {'state':'error','error':'broken mapping config'}]
        for payload in payloads:
            with self.subTest(payload=payload):
                class Mapping(fixtures.FakeComponent):
                    def dispatch(self,prefix,args):
                        if args.get('action')=='info':return dict(payload)
                        return super().dispatch(prefix,args)
                runtime=fixtures.FakeRuntime()
                plugin=ControlledSemanticSpatialPlugin({},'ubuntu',None,runtime=runtime,mapping_plugin=Mapping('mapping'),
                    planning_plugin=fixtures.FakeComponent('planning'),semantic_plugin=fixtures.FakeComponent('semantic'))
                r=plugin.dispatch(plugin.PREFIX,{'action':'start','input_bindings':fixtures._external_bindings_for_start()})
                self.assertEqual(r['error_code'],'invalid_config');self.assertIsNone(runtime.start_kwargs)

    def test_adapter_configuration_failure_rolls_back_unified_runtime(self):
        class Backend:
            def info(self):return {'state':'ready','backend':'fast_livo2_ros_topic','bridge_subscribers':1}
            def execute(self,action,args):return {'status':'error','error_code':'test_filter_rejected','error':'rejected'}
            def stop(self):pass
        runtime=fixtures.FakeRuntime();mapping=FastLivo2Plugin({},None,backend=Backend())
        plugin=ControlledSemanticSpatialPlugin({},'ubuntu',None,runtime=runtime,mapping_plugin=mapping,
            planning_plugin=fixtures.FakeComponent('planning'),semantic_plugin=fixtures.FakeComponent('semantic'))
        r=plugin.dispatch(plugin.PREFIX,{'action':'start','input_bindings':fixtures._external_bindings_for_start()})
        self.assertEqual(r['state'],'error');self.assertFalse(runtime.started);self.assertEqual(runtime.stop_calls,1)

    def test_runtime_rejects_bad_band_before_popen_and_live_band_changes_without_restart(self):
        factory=mock.Mock(side_effect=lambda command,**kw:fixtures.ControlledSemanticSpatialRuntimeTest.Process(command))
        runtime=ControlledSemanticSpatialRuntime(popen_factory=factory,startup_grace_sec=0)
        for lo,hi in ((True,.3),(-4,.3),(-.5,float('nan')),(.3,.3)):
            with self.assertRaises(ValueError):runtime.start(obstacle_min_height_m=lo,obstacle_max_height_m=hi)
        factory.assert_not_called()
        with mock.patch.object(runtime,'_start_descendant_monitor'):
            self.assertEqual(runtime.start(obstacle_min_height_m=-.5,obstacle_max_height_m=.3)['launch_obstacle_height_range_m'],[-.5,.3])
            self.assertTrue(runtime.start(obstacle_min_height_m=-.5,obstacle_max_height_m=.3)['already_started'])
            with mock.patch.object(runtime,'_stop_locked') as stop:
                with self.assertRaises(ValueError):runtime.start(obstacle_min_height_m=-.2,obstacle_max_height_m=.3)
                stop.assert_not_called()
        self.assertEqual(factory.call_count,2)


class Capture:
    def __init__(self,*args,**kwargs):self.args=args;self.kwargs=kwargs
class Config(Capture):
    @property
    def name(self):return self.args[0]
class Parameter(Capture):pass
class YamlHarness(Capture):
    """ROS-free configuration harness, NOT claimed to be ROS RewrittenYaml."""
    def convert(self,v):
        try:return float(v)
        except (TypeError,ValueError):return v
    def substitute_params(self,data,rewrites):
        def walk(node):
            for key,value in list(node.items()):
                if isinstance(value,dict):walk(value)
                elif key in rewrites:node[key]=self.convert(rewrites[key])
        walk(data)


def load_launch(path):
    module_attributes={
        'launch':{'LaunchDescription':Capture},
        'launch.actions':{k:Capture for k in ('DeclareLaunchArgument','GroupAction','IncludeLaunchDescription')},
        'launch.launch_description_sources':{'PythonLaunchDescriptionSource':Capture},
        'launch.substitutions':{'LaunchConfiguration':Config},
        'launch_ros.actions':{'Node':Capture,'SetRemap':Capture},
        'launch_ros.parameter_descriptions':{'ParameterValue':Parameter},
        'ament_index_python.packages':{'get_package_share_directory':lambda name:'/share/'+name},
        'nav2_common.launch':{'RewrittenYaml':YamlHarness},
    }
    modules={}
    for name,attrs in module_attributes.items():
        for i in range(1,len(name.split('.'))+1):
            prefix='.'.join(name.split('.')[:i]);modules.setdefault(prefix,types.ModuleType(prefix))
        modules[name].__dict__.update(attrs)
    with mock.patch.dict(sys.modules,modules):return runpy.run_path(str(path))


def captured_objects(value):
    if isinstance(value,Capture):
        yield value
        for child in (*value.args,*value.kwargs.values()):yield from captured_objects(child)
    elif isinstance(value,dict):
        for child in value.values():yield from captured_objects(child)
    elif isinstance(value,(list,tuple)) or type(value).__name__=='dict_items':
        for child in value:yield from captured_objects(child)


class HeightBandLaunchTest(unittest.TestCase):
    def test_both_launches_require_explicit_band_and_adapter_receives_float_parameters(self):
        path=PLUGIN/'runtime/fast_livo2/launch/fast_livo2.launch.py'
        ns=load_launch(path);ld=ns['generate_launch_description']();objects=list(captured_objects(ld))
        adapter=next(x for x in objects if x.kwargs.get('executable')=='frame_adapter')
        merged={k:v for entry in adapter.kwargs['parameters'] for k,v in entry.items()}
        for name in ('obstacle_min_height_m','obstacle_max_height_m'):
            self.assertIsInstance(merged[name],Parameter)
            self.assertIs(merged[name].kwargs['value_type'],float)
            self.assertEqual(merged[name].args[0].name,name)
            declaration=next(x for x in objects if x.args==(name,))
            self.assertNotIn('default_value',declaration.kwargs)

    def test_nav2_include_uses_height_rewriter_and_repairs_legacy_yaml(self):
        ns=load_launch(NAV2/'launch/nav2.launch.py');ld=ns['generate_launch_description']();objects=list(captured_objects(ld))
        include=next(x for x in objects if 'launch_arguments' in x.kwargs)
        params=dict(include.kwargs['launch_arguments'])['params_file']
        self.assertIsInstance(params,ns['HeightBandRewrittenYaml'])
        self.assertTrue(params.kwargs['convert_types'])
        rewrites=params.kwargs['param_rewrites']
        self.assertEqual(rewrites['min_obstacle_height'].name,'obstacle_min_height_m')
        self.assertEqual(rewrites['max_obstacle_height'].name,'obstacle_max_height_m')
        data=template()
        for name in ('local_costmap','global_costmap'):
            layer=data[name][name]['ros__parameters']['obstacle_layer'];del layer['min_obstacle_height'],layer['max_obstacle_height']
        # Actual subclass + pure helper are exercised; only the launch/ROS base is a harness.
        params.substitute_params(data,{'min_obstacle_height':'-0.5','max_obstacle_height':'0.3','rotate_reengage_rad':'0.12','inflation_radius':'0.55'})
        self.assertEqual(limits(data),[(-.5,.3)]*4)
        self.assertEqual(data['controller_server']['ros__parameters']['FollowPath']['rotate_reengage_rad'],.12)
        for name in ('obstacle_min_height_m','obstacle_max_height_m'):
            declaration=next(x for x in objects if x.args==(name,))
            self.assertNotIn('default_value',declaration.kwargs)


class InflationRadiusLaunchTest(unittest.TestCase):
    def test_nondefault_radius_rewrites_only_two_float_leaves_with_legacy_support(self):
        ns = load_launch(NAV2 / 'launch/nav2.launch.py')
        objects = list(captured_objects(ns['generate_launch_description']()))
        params = dict(next(x for x in objects if 'launch_arguments' in x.kwargs).kwargs['launch_arguments'])['params_file']
        self.assertEqual(params.kwargs['param_rewrites']['inflation_radius'].name, 'inflation_radius_m')
        declaration = next(x for x in objects if x.args == ('inflation_radius_m',))
        self.assertNotIn('default_value', declaration.kwargs)
        for radius in (.45, .55, .8, 1):
            for legacy in (False, True):
                with self.subTest(radius=radius, legacy=legacy):
                    data = template()
                    before = copy.deepcopy(data)
                    data['unrelated'] = {'inflation_radius': 17.}
                    if legacy:
                        for name in ('local_costmap', 'global_costmap'):
                            del data[name][name]['ros__parameters']['inflation_layer']['inflation_radius']
                    params.substitute_params(data, {'min_obstacle_height': '-0.5', 'max_obstacle_height': '0.3',
                                                   'inflation_radius': str(radius)})
                    for name in ('local_costmap', 'global_costmap'):
                        p = data[name][name]['ros__parameters']
                        self.assertEqual(p['inflation_layer']['inflation_radius'], radius)
                        self.assertIs(type(p['inflation_layer']['inflation_radius']), float)
                        self.assertEqual(p['inflation_layer']['cost_scaling_factor'], before[name][name]['ros__parameters']['inflation_layer']['cost_scaling_factor'])
                        self.assertEqual(p['footprint'], before[name][name]['ros__parameters']['footprint'])
                    self.assertEqual(data['unrelated']['inflation_radius'], 17.)
                    self.assertEqual(limits(data), [(-.5, .3)] * 4)

    def test_invalid_radius_and_layers_fail_before_radius_mutation(self):
        from nav2.height_band import apply_inflation_radius
        for bad in (None, True, '0.8', 0, -.1, 2.01, float('nan'), float('inf'), 10**1000):
            data = template(); before = copy.deepcopy(data)
            with self.assertRaises(ValueError): apply_inflation_radius(data, bad)
            self.assertEqual(data, before)
        for mutate in (lambda p: p.pop('inflation_layer'),
                       lambda p: p['inflation_layer'].update(plugin='wrong::Layer'),
                       lambda p: p.update(plugins=['obstacle_layer'])):
            data = template()
            mutate(data['global_costmap']['global_costmap']['ros__parameters'])
            before = copy.deepcopy(data)
            with self.assertRaises(ValueError): apply_inflation_radius(data, .8)
            self.assertEqual(data, before)

    def test_runtime_launch_arguments_and_live_change_guard(self):
        factory = mock.Mock(side_effect=lambda command, **kw: fixtures.ControlledSemanticSpatialRuntimeTest.Process(command))
        runtime = ControlledSemanticSpatialRuntime(popen_factory=factory, startup_grace_sec=0)
        for bad in (None, True, '0.8', 0, -.1, 2.01, float('nan'), float('inf'), 10**1000):
            with self.assertRaises(ValueError): runtime.start(inflation_radius_m=bad)
        factory.assert_not_called()
        for radius in (.45, .55, .8, 1):
            args = ControlledSemanticSpatialRuntime._launch_arguments('ubuntu', {}, inflation_radius_m=radius)
            self.assertIn('inflation_radius_m:=' + str(float(radius)), args['nav2'])
            self.assertFalse(any('inflation_radius' in value for value in args['fast_livo2']))
        with mock.patch.object(runtime, '_start_descendant_monitor'):
            self.assertEqual(runtime.start(inflation_radius_m=.8)['launch_inflation_radius_m'], .8)
            self.assertTrue(runtime.start(inflation_radius_m=.8)['already_started'])
            with mock.patch.object(runtime, '_stop_locked') as stop:
                with self.assertRaises(ValueError): runtime.start(inflation_radius_m=.45)
                stop.assert_not_called()
        self.assertEqual(factory.call_count, 2)


if __name__=='__main__':unittest.main()

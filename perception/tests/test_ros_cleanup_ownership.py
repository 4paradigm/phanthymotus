"""Deterministic public-dispatch regressions for every shared cleanup owner.

ROS, models and network access use the existing plugin test fakes. Events
select race windows; no timing sleeps, real hardware or SDK calls are used.
"""
import sys
import threading

import pytest
import vision_stubs
from vision_stubs import _FakeExecutor, _wait_until

KINDS = ['ocr', 'pose', 'face_recognition', 'soundevent', 'barcode',
         'visual_depth', 'echo_cancellation']

def build(kind, monkeypatch, tmp_path):
    if kind == 'pose':
        import test_pose_plugin as t
        plugin, executor = t._plugin()
    elif kind == 'visual_depth':
        import test_visual_depth_plugin as t
        plugin, executor = t._plugin(model=t._FakeModel())
    elif kind == 'ocr':
        import test_ocr_plugin as t
        plugin, executor, _ = t._make_plugin(monkeypatch)
    elif kind == 'barcode':
        import test_barcode_plugin as t
        plugin, executor, _ = t._make_plugin(monkeypatch)
    elif kind == 'face_recognition':
        import test_face_plugin as t
        t._tmp_models.__wrapped__(monkeypatch, tmp_path)
        monkeypatch.setattr(t.face_plugin, '_build_engine', t._EngineProbe(tmp_path))
        executor = _FakeExecutor()
        plugin = t.face_plugin.FaceRecognitionPlugin(t._base_cfg(tmp_path), executor)
    elif kind == 'soundevent':
        import test_soundevent_plugin as t
        monkeypatch.setattr(t.soundevent, '_build_model', t._ModelBuilderProbe())
        executor = _FakeExecutor()
        plugin = t.soundevent.SoundEventPlugin({}, executor)
    elif kind == 'echo_cancellation':
        import test_echo_cancellation_plugin as t
        executor = _FakeExecutor()
        plugin = t.ec.EchoCancellationPlugin({}, executor, tts=t._FakeTTS())
    else:
        import test_vop_plugin as t
        plugin, executor = t._plugin(model=t._FakeModel())
    return plugin, executor



def owned(plugin, node):
    return node in plugin._nodes.values() or any(
        node in values for values in plugin._draining_nodes.values())


def start(plugin, kind, key='audit', topic='/audit/input'):
    return plugin.dispatch(kind, {'action': 'start', 'instance_id': key,
                                 'input_topic': topic})


def stop(plugin, kind, key='audit'):
    return plugin.dispatch(kind, {'action': 'stop', 'instance_id': key})


def info(plugin, kind, key='audit'):
    return plugin.dispatch(kind, {'action': 'info', 'instance_id': key})


@pytest.fixture(params=KINDS)
def card(request, monkeypatch, tmp_path):
    kind = request.param
    plugin, executor = build(kind, monkeypatch, tmp_path)
    # rclpy exposes membership. The old external audit also runs unchanged
    # against executors without this optional API.
    executor.get_nodes = lambda: list(executor.nodes)
    start(plugin, kind)
    assert _wait_until(lambda: info(plugin, kind)['state'] == 'running')
    yield kind, plugin, executor
    assert plugin.dispatch(kind, {'action': 'stop'})['state'] == 'idle'
    assert not plugin._nodes and not plugin._draining_nodes and not executor.nodes


@pytest.mark.parametrize('failure', ['remove_raise', 'remove_false', 'remove_silent',
                                    'destroy_raise', 'destroy_false'])
def test_failure_retains_stages_blocks_replacement_and_stop_retries(card, monkeypatch, failure):
    kind, plugin, executor = card
    node = plugin._nodes['audit']
    remove, destroy = executor.remove_node, node.destroy_node
    fail = [True]
    removes, destroys = [], []

    def detach(target):
        removes.append(target)
        if fail[0] and failure == 'remove_raise':
            raise RuntimeError('detach failed')
        if fail[0] and failure == 'remove_false':
            return False
        if fail[0] and failure == 'remove_silent':
            return None
        return remove(target)

    def release():
        destroys.append(node)
        assert node not in executor.nodes
        if fail[0] and failure == 'destroy_raise':
            raise RuntimeError('destroy failed')
        if fail[0] and failure == 'destroy_false':
            return False
        return destroy()

    monkeypatch.setattr(executor, 'remove_node', detach)
    monkeypatch.setattr(node, 'destroy_node', release)
    try:
        for _ in range(2):
            result = stop(plugin, kind)
            assert result['state'] == 'failed' and result['error']
            assert info(plugin, kind)['state'] == 'failed'
            assert plugin.dispatch(kind, {'action': 'info'})['state'] == 'failed'
            assert owned(plugin, node) and not node.destroyed
            progress = node._lifecycle_disposal
            assert progress.removed == failure.startswith('destroy')
            assert not progress.destroyed and progress.error
            assert (node in executor.nodes) == failure.startswith('remove')
            before = len(vision_stubs._FakeNode.instances)
            assert start(plugin, kind, topic='/audit/replacement')['state'] == 'failed'
            assert len(vision_stubs._FakeNode.instances) == before
        assert len(removes) == (2 if failure.startswith('remove') else 1)
        assert len(destroys) == (0 if failure.startswith('remove') else 2)
    finally:
        fail[0] = False
    assert stop(plugin, kind)['state'] == 'idle'
    assert not owned(plugin, node) and node.destroyed and node not in executor.nodes
    count = len(destroys)
    assert stop(plugin, kind)['state'] == 'idle' and len(destroys) == count
    assert node._lifecycle_disposal.complete
    assert start(plugin, kind, topic='/audit/replacement')['state'] == 'running'
    assert plugin._nodes['audit'] is not node


def thread_call(callback):
    result = []
    def run():
        result.append(callback())
    thread = threading.Thread(target=run)
    thread.start()
    return thread, result


def test_two_stops_serialize_cleanup_and_never_destroy_twice(card, monkeypatch):
    kind, plugin, executor = card
    node = plugin._nodes['audit']
    entered, release = threading.Event(), threading.Event()
    remove, destroy = executor.remove_node, node.destroy_node
    calls = []
    def detach(target):
        calls.append('remove')
        entered.set()
        assert release.wait(5)
        return remove(target)
    def dispose():
        calls.append('destroy')
        return destroy()
    monkeypatch.setattr(executor, 'remove_node', detach)
    monkeypatch.setattr(node, 'destroy_node', dispose)
    thread, results = thread_call(lambda: stop(plugin, kind))
    try:
        assert entered.wait(3)
        assert stop(plugin, kind)['state'] == 'stopping'
        assert start(plugin, kind)['state'] == 'stopping'
        assert owned(plugin, node) and not node.destroyed
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and results[0]['state'] == 'idle'
    assert stop(plugin, kind)['state'] == 'idle'
    assert calls == ['remove', 'destroy']


@pytest.mark.parametrize('boundary', ['constructor', 'registration'])
def test_stop_cancels_inflight_start_before_replacement_is_allowed(card, monkeypatch, boundary):
    kind, plugin, executor = card
    previous = plugin._nodes['audit']
    cls = type(previous)
    assert stop(plugin, kind)['state'] == 'idle'
    entered, release = threading.Event(), threading.Event()
    created = []
    if boundary == 'constructor':
        def factory(*args, **kwargs):
            node = cls(*args, **kwargs)
            created.append(node)
            entered.set()
            assert release.wait(5)
            return node
        monkeypatch.setattr(sys.modules[cls.__module__], cls.__name__, factory)
    else:
        add = executor.add_node
        def register(node):
            add(node)
            created.append(node)
            entered.set()
            assert release.wait(5)
        monkeypatch.setattr(executor, 'add_node', register)
    thread, results = thread_call(lambda: start(plugin, kind))
    try:
        assert entered.wait(3)
        old_request = plugin._lifecycle.current('audit')
        assert stop(plugin, kind)['state'] == 'stopping'
        assert start(plugin, kind)['state'] == 'stopping'
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and results
    assert len(created) == 1 and created[0].destroyed
    assert not executor.nodes and not plugin._draining_nodes and not plugin._nodes
    assert info(plugin, kind)['state'] == 'idle'
    assert start(plugin, kind)['state'] == 'running'
    replacement = plugin._nodes['audit']
    assert plugin._lifecycle.current('audit').generation > old_request.generation
    def forbidden():
        pytest.fail('expired request constructed a node')
    assert plugin._lifecycle.start('audit', old_request, forbidden,
                                  plugin._lifecycle.retire_node)['state'] == 'idle'
    assert plugin._nodes['audit'] is replacement and not replacement.destroyed


@pytest.mark.parametrize('failure', ['raise', 'false'])
@pytest.mark.parametrize('membership_api', [True, False])
def test_partial_executor_registration_failure_retains_candidate(card, monkeypatch, failure, membership_api):
    kind, plugin, executor = card
    assert stop(plugin, kind)['state'] == 'idle'
    if not membership_api:
        monkeypatch.delattr(executor, 'get_nodes')
    add, remove = executor.add_node, executor.remove_node
    created = []
    def register(node):
        add(node)
        created.append(node)
        if failure == 'raise':
            raise RuntimeError('registration failed after insertion')
        return False
    def detach(node):
        raise RuntimeError('executor detach unavailable')
    monkeypatch.setattr(executor, 'add_node', register)
    monkeypatch.setattr(executor, 'remove_node', detach)
    try:
        assert start(plugin, kind)['state'] == 'failed'
        node = created[0]
        assert owned(plugin, node) and node in executor.nodes and not node.destroyed
        assert start(plugin, kind)['state'] == 'failed'
        assert len(created) == 1
    finally:
        monkeypatch.setattr(executor, 'add_node', add)
        monkeypatch.setattr(executor, 'remove_node', remove)
    assert stop(plugin, kind)['state'] == 'idle'
    assert node.destroyed and not executor.nodes


@pytest.mark.parametrize('kind', ['ocr', 'face_recognition', 'soundevent', 'pose', 'visual_depth'])
def test_cancelled_model_load_only_activates_the_new_generation(kind, monkeypatch, tmp_path):
    plugin, executor = build(kind, monkeypatch, tmp_path)
    entered, release = threading.Event(), threading.Event()
    if kind in ('pose', 'visual_depth'):
        model, plugin._model = plugin._model, None
        def load():
            entered.set()
            assert release.wait(5)
            plugin._model = model
        monkeypatch.setattr(plugin, '_ensure_model', load)
    else:
        modules = {'ocr': ('plugins.ocr', '_build_ocr_adapter'),
                   'face_recognition': ('plugins.face', '_build_engine'),
                   'soundevent': ('plugins.soundevent', '_build_model')}
        name, attribute = modules[kind]
        module = sys.modules[name]
        builder = getattr(module, attribute)
        def load(*args, **kwargs):
            entered.set()
            assert release.wait(5)
            return builder(*args, **kwargs)
        monkeypatch.setattr(module, attribute, load)
    try:
        assert start(plugin, kind)['state'] == 'loading'
        assert entered.wait(3)
        assert stop(plugin, kind)['state'] == 'idle'
        assert start(plugin, kind, key='new', topic='/audit/new')['state'] == 'loading'
    finally:
        release.set()
    try:
        assert _wait_until(lambda: info(plugin, kind, key='new')['state'] == 'running')
        assert set(plugin._nodes) == {'new'}
        assert info(plugin, kind)['state'] == 'idle'
        assert len(executor.nodes) == 1
    finally:
        assert plugin.dispatch(kind, {'action': 'stop'})['state'] == 'idle'


@pytest.mark.parametrize('kind', ['pose', 'visual_depth'])
def test_native_worker_timeout_retains_thread_and_node(kind, monkeypatch, tmp_path):
    plugin, executor = build(kind, monkeypatch, tmp_path)
    executor.get_nodes = lambda: list(executor.nodes)
    assert start(plugin, kind)['state'] == 'running'
    node = plugin._nodes['audit']
    node.request_stop()
    node._worker.join(3)
    assert not node._worker.is_alive()
    class FinishingWorker:
        alive = True
        def is_alive(self):
            return self.alive
        def join(self, timeout=None):
            assert timeout == 3.0
    worker = node._worker = FinishingWorker()
    try:
        assert stop(plugin, kind)['state'] == 'stopping'
        assert node._worker is worker and owned(plugin, node)
        assert node in executor.nodes and not node.destroyed
        assert start(plugin, kind)['state'] == 'stopping'
    finally:
        worker.alive = False
    assert stop(plugin, kind)['state'] == 'idle'
    assert node.destroyed and not executor.nodes


def test_old_stop_cannot_retire_a_new_request(monkeypatch, tmp_path):
    plugin, executor = build('barcode', monkeypatch, tmp_path)
    lifecycle = plugin._lifecycle
    old_request, blocked = lifecycle.begin('audit', '/audit/old')
    assert not blocked
    entered, release = threading.Event(), threading.Event()
    operation = lifecycle._operation
    old_thread = [None]
    def paused_operation(key):
        if threading.current_thread() is old_thread[0]:
            entered.set()
            assert release.wait(5)
        return operation(key)
    monkeypatch.setattr(lifecycle, '_operation', paused_operation)
    results = []
    thread = threading.Thread(target=lambda: results.append(stop(plugin, 'barcode')))
    old_thread[0] = thread
    thread.start()
    try:
        assert entered.wait(3)
        assert old_request.cancelled
        assert start(plugin, 'barcode', topic='/audit/new')['state'] == 'running'
        replacement = plugin._nodes['audit']
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and results
    assert plugin._nodes['audit'] is replacement and replacement in executor.nodes
    assert not replacement.destroyed
    assert stop(plugin, 'barcode')['state'] == 'idle'


@pytest.mark.parametrize('kind', ['ocr', 'face_recognition', 'echo_cancellation'])
def test_model_or_reference_rebuild_surfaces_cleanup_failure(kind, monkeypatch, tmp_path):
    plugin, executor = build(kind, monkeypatch, tmp_path)
    executor.get_nodes = lambda: list(executor.nodes)
    start(plugin, kind)
    assert _wait_until(lambda: info(plugin, kind)['state'] == 'running')
    node = plugin._nodes['audit']
    remove = executor.remove_node
    def fail(target):
        raise RuntimeError('rebuild detach unavailable')
    monkeypatch.setattr(executor, 'remove_node', fail)
    configs = {'ocr': {'model_dir': '/models/different'},
               'face_recognition': {'model': 'buffalo_l'},
               'echo_cancellation': {'robot_voice_topic': '/other/ref'}}
    try:
        result = plugin.dispatch(kind, {'action': 'config', **configs[kind]})
        assert result['state'] == 'failed'
        assert owned(plugin, node) and not node.destroyed
        assert start(plugin, kind)['state'] == 'failed'
    finally:
        monkeypatch.setattr(executor, 'remove_node', remove)
    assert stop(plugin, kind)['state'] == 'idle'
    assert node.destroyed and not executor.nodes


def test_failed_node_start_keeps_cleanup_owner_until_stop_recovers(card, monkeypatch):
    kind, plugin, executor = card
    cls = type(plugin._nodes['audit'])
    assert stop(plugin, kind)['state'] == 'idle'
    remove = executor.remove_node
    def broken_start(node):
        raise RuntimeError('node start failed')
    def broken_remove(node):
        raise RuntimeError('startup cleanup failed')
    monkeypatch.setattr(cls, 'start', broken_start)
    monkeypatch.setattr(executor, 'remove_node', broken_remove)
    try:
        assert start(plugin, kind)['state'] == 'failed'
        assert len(executor.nodes) == 1
        node = executor.nodes[0]
        assert owned(plugin, node) and not node.destroyed
        assert info(plugin, kind)['state'] == 'failed'
        assert start(plugin, kind)['state'] == 'failed'
        assert executor.nodes == [node]
    finally:
        monkeypatch.setattr(executor, 'remove_node', remove)
    assert stop(plugin, kind)['state'] == 'idle'
    assert node.destroyed and not executor.nodes


def test_depthart_shutdown_does_not_close_model_before_failed_cleanup_recovers(monkeypatch, tmp_path):
    plugin, executor = build('visual_depth', monkeypatch, tmp_path)
    executor.get_nodes = lambda: list(executor.nodes)
    assert start(plugin, 'visual_depth')['state'] == 'running'
    node = plugin._nodes['audit']
    # Exercise the existing DepthART-only model-close branch after a host-side
    # fake node is running; no DepthART library or native engine is loaded.
    plugin._backend = 'depthart'
    model = plugin._model
    closed = []
    monkeypatch.setattr(model, 'close', lambda: closed.append(True), raising=False)
    remove = executor.remove_node
    def fail(target):
        raise RuntimeError('shutdown detach unavailable')
    monkeypatch.setattr(executor, 'remove_node', fail)
    try:
        assert plugin.shutdown()['state'] == 'failed'
        assert plugin._model is model and not closed
        assert owned(plugin, node) and not node.destroyed
        assert start(plugin, 'visual_depth')['state'] in ('error', 'failed')
    finally:
        monkeypatch.setattr(executor, 'remove_node', remove)
    assert plugin.shutdown()['state'] == 'idle'
    assert closed == [True] and plugin._model is None
    assert node.destroyed and not executor.nodes
    assert plugin.shutdown()['state'] == 'idle' and closed == [True]


@pytest.mark.parametrize('kind', ['ocr', 'face_recognition'])
def test_model_config_during_loading_revokes_old_request_before_replay(kind, monkeypatch, tmp_path):
    plugin, executor = build(kind, monkeypatch, tmp_path)
    module = sys.modules['plugins.ocr' if kind == 'ocr' else 'plugins.face']
    attribute = '_build_ocr_adapter' if kind == 'ocr' else '_build_engine'
    original = getattr(module, attribute)
    entered, release = threading.Event(), threading.Event()
    calls = []
    def builder(*args, **kwargs):
        calls.append(args[0].copy())
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, attribute, builder)
    try:
        assert start(plugin, kind)['state'] == 'loading'
        assert entered.wait(3)
        old_request = plugin._lifecycle.current('audit')
        cfg = {'model_dir': '/models/new-profile'} if kind == 'ocr' else {'det_thresh': 0.7}
        result = plugin.dispatch(kind, {'action': 'config', **cfg})
        assert result['status'] == 'configured'
        assert old_request.cancelled
    finally:
        release.set()
    try:
        assert _wait_until(lambda: info(plugin, kind)['state'] == 'running')
        assert len(calls) == 2 and len(executor.nodes) == 1
        request = plugin._lifecycle.current('audit')
        assert request is not old_request and request.generation > old_request.generation
    finally:
        assert plugin.dispatch(kind, {'action': 'stop'})['state'] == 'idle'


def test_shutdown_retries_failed_cleanup_and_rejects_new_starts(card, monkeypatch):
    kind, plugin, executor = card
    node = plugin._nodes['audit']
    remove = executor.remove_node
    def fail(target):
        raise RuntimeError('shutdown detach failed')
    monkeypatch.setattr(executor, 'remove_node', fail)
    try:
        assert plugin.shutdown()['state'] == 'failed'
        assert owned(plugin, node) and not node.destroyed
        assert start(plugin, kind)['state'] == 'failed'
    finally:
        monkeypatch.setattr(executor, 'remove_node', remove)
    assert plugin.shutdown()['state'] == 'idle'
    assert node.destroyed and not executor.nodes
    assert plugin.shutdown()['state'] == 'idle'
    assert start(plugin, kind)['state'] in ('error', 'failed')


def test_depth_backend_switch_keeps_failed_resources_model_alive(monkeypatch, tmp_path):
    plugin, executor = build('visual_depth', monkeypatch, tmp_path)
    executor.get_nodes = lambda: list(executor.nodes)
    assert start(plugin, 'visual_depth')['state'] == 'running'
    node, model = plugin._nodes['audit'], plugin._model
    closed = []
    monkeypatch.setattr(model, 'close', lambda: closed.append(True), raising=False)
    remove = executor.remove_node
    def fail(target):
        raise RuntimeError('backend switch detach unavailable')
    monkeypatch.setattr(executor, 'remove_node', fail)
    try:
        assert stop(plugin, 'visual_depth')['state'] == 'failed'
        result = plugin.dispatch('visual_depth', {'action': 'config', 'backend': 'depthart'})
        assert result['state'] == 'failed'
        assert plugin._backend == 'yolo' and plugin._model is model and not closed
        assert owned(plugin, node) and not node.destroyed
    finally:
        monkeypatch.setattr(executor, 'remove_node', remove)
    assert stop(plugin, 'visual_depth')['state'] == 'idle'
    assert plugin.dispatch('visual_depth', {'action': 'config', 'backend': 'depthart'})['status'] == 'configured'
    assert closed == [True] and node.destroyed


def test_live_depthart_rebind_is_rejected_before_retiring_old_owner(monkeypatch, tmp_path):
    plugin, executor = build('visual_depth', monkeypatch, tmp_path)
    assert start(plugin, 'visual_depth')['state'] == 'running'
    node, model = plugin._nodes['audit'], plugin._model
    from test_depthart_runtime import camera
    plugin._backend = 'depthart'
    plugin._upstream_camera['audit'] = camera()
    monkeypatch.setattr(model, 'for_camera', lambda binding: model, raising=False)
    request = plugin._lifecycle.current('audit')
    try:
        result = plugin.dispatch('visual_depth', {'action': 'start', 'instance_id': 'audit',
                    'input_topic': '/audit/other', 'camera_info': {'/audit/other': camera()}})
        assert plugin._nodes['audit'] is node and not node.destroyed
        assert result['state'] == 'error'
        assert plugin._lifecycle.current('audit') is request and not request.cancelled
        assert not node._stop_event.is_set()
    finally:
        assert plugin.dispatch('visual_depth', {'action': 'stop'})['state'] == 'idle'


def test_repeated_live_start_is_idempotent(card):
    kind, plugin, executor = card
    node = plugin._nodes['audit']
    request = plugin._lifecycle.current('audit')
    for _ in range(3):
        assert start(plugin, kind)['state'] == 'running'
        assert plugin._nodes['audit'] is node and executor.nodes == [node]
        assert plugin._lifecycle.current('audit') is request
        assert not node.destroyed

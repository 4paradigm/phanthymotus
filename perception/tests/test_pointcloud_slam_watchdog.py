from plugins.pointcloud_slam_watchdog import BackendProgress


def test_idle_input_does_not_start_watchdog():
    p=BackendProgress(timeout_s=2)
    assert p.status(50)['state']=='idle'


def test_stationary_keyframe_skip_is_not_stall():
    p=BackendProgress(timeout_s=2)
    p.forward(1,1.);p.graph(1,1.1,{1},[])
    p.forward(2,2.);p.activity(2,2.1)
    assert p.status(20)['state']=='responsive_without_new_nodes'


def test_pending_input_without_native_activity_reports_suspected_stall():
    p=BackendProgress(timeout_s=2);p.forward(1,1.)
    assert p.status(3.1)['state']=='suspected_stall'
    assert p.status(3.1)['automatic_restart_allowed'] is False


def test_old_activity_cannot_hide_pending_and_restart_budget_is_zero():
    p=BackendProgress(timeout_s=2)
    p.forward(10,1.);p.activity(9,2.)
    assert p.status(4.1)['state']=='suspected_stall'
    assert p.status(4.1)['restart_budget']==0


def test_result_received_before_forward_accounting_does_not_start_stall_timer():
    p=BackendProgress(timeout_s=2)
    p.activity(10,1.);p.forward(10,1.1)
    assert p.status(10.)['state']=='responsive_without_new_nodes'

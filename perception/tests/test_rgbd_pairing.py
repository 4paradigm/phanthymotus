from dataclasses import replace
from types import SimpleNamespace as NS
from plugins.spatial_time import TimePolicy
from plugins.rgbd_pairing import RgbdPairer


def context(stamp=1_000_000_000, *, receipt=False, epoch='one', capture=None):
    source=dict(stamp_ns=stamp,clock_id='camera',source_epoch=epoch,
                timestamp_semantics='source_capture_mapped',source_capture_id=capture)
    if receipt:
        source.pop('timestamp_semantics');source.pop('stamp_ns')
    return TimePolicy('shared').select(source,stamp+10,stamp)


def msg():return NS(data=b'abc')


def test_shared_capture_id_retains_independent_observation_identities():
    p=RgbdPairer();a=context(capture='1');b=context(1_001_000_000,capture='1')
    assert p.push('color',msg(),a) is None
    pair=p.push('depth',msg(),b)
    assert pair.method=='capture_id' and pair.skew_ns==1_000_000
    assert pair.color_context is a and pair.depth_context is b
    assert pair.association_id != a.observation_id


def test_receipt_pair_retains_separate_receipt_contexts():
    p=RgbdPairer();a=context(receipt=True);b=context(1_001_000_000,receipt=True)
    p.push('color',msg(),a);pair=p.push('depth',msg(),b)
    assert pair.method=='receipt_estimated'
    assert pair.color_context.source_capture_id is None and pair.depth_context.source_capture_id is None


def test_epoch_and_time_basis_labels_do_not_prevent_pairing():
    for kind,other in (('depth',context(epoch='two')),('color',context(receipt=True))):
        p=RgbdPairer();p.push('color',msg(),context())
        pair = p.push(kind,msg(),other)
        assert (pair is not None) == (kind == 'depth')


def test_count_byte_and_ingress_age_limit_pending_frames():
    p=RgbdPairer(max_pending=2,max_bytes=6,max_wait_ns=100)
    for i in range(5):p.push('color',msg(),context(1_000_000_000+i))
    assert p.pending_count<=2 and p.pending_bytes<=6
    assert p.push('depth',msg(),context(1_000_000_200)) is None
    assert p.pending_count == 1 and p.dropped_by_side == {'color': 5, 'depth': 0}


def test_repeated_stamps_and_different_capture_ids_still_pair():
    p=RgbdPairer();p.push('color',msg(),context(capture='a'))
    assert p.push('depth',msg(),context(capture='b')) is not None
    assert p.push('color',msg(),context(capture='a')) is None


def test_source_time_and_geometry_version_labels_do_not_block_pairing():
    p=RgbdPairer();p.push('color',msg(),context())
    pair=p.push('depth',msg(),context(1_000_000_001))
    assert pair.method=='nearest_capture'
    p=RgbdPairer();p.push('color',msg(),replace(context(),geometry_version='a'))
    assert p.push('depth',msg(),replace(context(),geometry_version='b')) is not None


def test_optional_epoch_on_one_stream_is_unknown_not_a_reset():
    p=RgbdPairer();a=replace(context(),source_epoch=None);b=context()
    p.push('color',msg(),a)
    pair=p.push('depth',msg(),b)
    assert pair.color_context.source_epoch is None and pair.depth_context.source_epoch=='one'
    p.push('color',msg(),replace(a,observation_stamp_ns=a.observation_stamp_ns+1))
    assert p.push('depth',msg(),context(1_000_000_001,epoch='two')) is not None


def test_one_missing_capture_preserves_the_other_source_time():
    p=RgbdPairer();a=context();b=context(1_001_000_000,receipt=True)
    p.push('color',msg(),a)
    pair=p.push('depth',msg(),b)
    assert pair.method=='receipt_estimated'
    assert pair.color_context.time_basis=='source_capture'
    assert pair.color_time.time_basis=='receipt_estimated'
    assert pair[0][0]==a.ingress_realtime_ns
    assert pair.color_time.parent_observation_ids == (a.observation_id,)
    assert pair.color_time.original_stamp==a.original_stamp
    assert pair.depth_context.source_capture_id is None


def test_buffered_images_choose_closest_available_capture_after_one_stream_drops_frames():
    pairer = RgbdPairer()
    old, current = context(1_000_000_000), context(1_066_000_000)
    pairer.push('color', msg(), old)
    pairer.push('color', msg(), current)
    pair = pairer.push('depth', msg(), context(1_066_010_000))
    assert pair.color_context is current
    assert pair.skew_ns == 10_000 and pair.method == 'nearest_capture'
    assert pairer.dropped == 1 and pairer.pending_count == 0


def test_closest_pair_preserves_later_candidates_and_breaks_ties_by_arrival():
    pairer = RgbdPairer()
    first, second = context(100), context(200)
    pairer.push('depth', msg(), first)
    pairer.push('depth', msg(), second)
    assert pairer.push('color', msg(), context(150)).depth_context is first
    assert pairer.push('color', msg(), context(3000)).depth_context is second

def test_one_far_capture_does_not_consume_its_partner_or_fall_back_to_near_receipts():
    pairer = RgbdPairer()
    old = context()
    depth = replace(context(1_300_000_000), ingress_monotonic_ns=old.ingress_monotonic_ns+1,
                    ingress_realtime_ns=old.ingress_realtime_ns+1)
    assert pairer.push('color', msg(), old) is None
    assert pairer.push('depth', msg(), depth) is None
    current = context(1_301_000_000)
    pair = pairer.push('color', msg(), current)
    assert pair.color_context is current and pair.depth_context is depth
    assert pairer.last_skew_ns == pair.skew_ns == 1_000_000
    assert pairer.dropped_by_side == {'color': 1, 'depth': 0}


def test_capture_identity_prefers_correct_frame_only_within_skew_bounds():
    pairer = RgbdPairer()
    same = context(1_000_000_000, capture='7')
    pairer.push('color', msg(), same)
    pairer.push('color', msg(), context(1_006_000_000, capture='8'))
    assert pairer.push('depth', msg(), context(1_006_000_000, capture='7')).color_context is same
    assert pairer.push('depth', msg(), context(1_100_000_000, capture='8')) is None


def test_different_capture_clocks_use_bounded_common_receipts():
    pairer = RgbdPairer()
    a = context()
    b = replace(context(50), normalized_clock_id='depth_clock',
                ingress_realtime_ns=a.ingress_realtime_ns+30_000_000,
                ingress_monotonic_ns=a.ingress_monotonic_ns+30_000_000)
    pairer.push('color', msg(), a)
    assert pairer.push('depth', msg(), b) is None
    near = replace(a, ingress_realtime_ns=b.ingress_realtime_ns+123,
                   ingress_monotonic_ns=b.ingress_monotonic_ns+123)
    pair = pairer.push('color', msg(), near)
    assert pair.method == 'receipt_estimated' and pair.skew_ns == 123
    assert pair.color_time.time_basis == pair.depth_time.time_basis == 'receipt_estimated'
    assert pair.color_context is near and pair.depth_context is b


def test_unscoped_capture_ids_do_not_override_closer_observation():
    pairer = RgbdPairer()
    first = context(1_000_000_000, epoch=None, capture='5')
    closer = context(1_008_000_000, epoch=None, capture='6')
    pairer.push('color', msg(), first)
    pairer.push('color', msg(), closer)
    pair = pairer.push('depth', msg(), context(1_008_000_000, epoch=None, capture='5'))
    assert pair.color_context is closer and pair.method == 'nearest_capture'


def test_receipt_fallback_rejects_distant_arrivals_across_a_wall_clock_step():
    pairer = RgbdPairer()
    first = context(receipt=True)
    second = replace(first, ingress_monotonic_ns=first.ingress_monotonic_ns+200_000_000)
    pairer.push('color', msg(), first)
    assert pairer.push('depth', msg(), second) is None

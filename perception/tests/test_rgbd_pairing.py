from dataclasses import replace
from types import SimpleNamespace as NS
import pytest
from plugins.spatial_time import TimePolicy
from plugins.rgbd_pairing import RgbdPairer


def context(stamp=1_000_000_000, *, receipt=False, epoch='one', capture=None):
    source=dict(stamp_ns=stamp,clock_id='camera',source_epoch=epoch,
                timestamp_semantics='source_capture_mapped',source_capture_id=capture)
    if receipt:source.pop('timestamp_semantics')
    return TimePolicy('shared').select(source,stamp+10,stamp)


def msg():return NS(data=b'abc')


def test_source_identity_pair_is_exact():
    p=RgbdPairer();a=context(capture='1');b=context(1_001_000_000,capture='1')
    assert p.push('color',msg(),a) is None
    pair=p.push('depth',msg(),b)
    assert pair.method=='source_identity' and pair.skew_ns==1_000_000
    assert pair.color_context is a and pair.depth_context is b
    assert pair.association_id != a.observation_id


def test_arrival_pair_is_approximate_with_separate_source_ids():
    p=RgbdPairer();a=context(receipt=True);b=context(1_001_000_000,receipt=True)
    p.push('color',msg(),a);pair=p.push('depth',msg(),b)
    assert pair.method=='arrival_time'
    assert pair.color_context.source_capture_id is None and pair.depth_context.source_capture_id is None


def test_old_epoch_never_pairs_and_time_basis_cannot_silently_change():
    for other in (context(epoch='two'),context(receipt=True)):
        p=RgbdPairer();p.push('color',msg(),context())
        with pytest.raises(ValueError,match='context changed'):p.push('depth',msg(),other)
        assert p.pending_count==0


def test_timeout_drops_unmatched_frame_and_limits_bytes():
    p=RgbdPairer(max_pending=2,max_bytes=6,max_wait_ns=100)
    for i in range(5):p.push('color',msg(),context(1_000_000_000+i))
    assert p.pending_count<=2 and p.pending_bytes<=6
    assert p.push('depth',msg(),context(1_000_000_200)) is None
    assert p.pending_count==1 and p.dropped>=5


def test_known_repeated_source_never_falls_back_and_ids_must_agree():
    p=RgbdPairer();p.push('color',msg(),context(capture='a'))
    assert p.push('depth',msg(),context(capture='b')) is None
    with pytest.raises(ValueError):p.push('color',msg(),context(capture='a'))


def test_source_time_and_geometry_version_conflict():
    p=RgbdPairer();p.push('color',msg(),context())
    pair=p.push('depth',msg(),context(1_000_000_001))
    assert pair.method=='source_time'
    p=RgbdPairer();p.push('color',msg(),replace(context(),geometry_version='a'))
    with pytest.raises(ValueError):p.push('depth',msg(),replace(context(),geometry_version='b'))

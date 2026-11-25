import hashlib
import struct
from dataclasses import replace
import pytest
import numpy as np
from test_pointcloud_slam_ros import cloud
from plugins.spatial_time import TimePolicy
from plugins.pointcloud_normalization import normalize_cloud


def ctx():return TimePolicy('receipt').select({},1_000_000_000,100)


def test_big_endian_xyz_matches_little_endian_and_output_digest_changes():
    source=cloud();source.is_bigendian=True
    source.data=struct.pack('>ffffff',1,2,3,.5,.001,42)
    digest=hashlib.sha256(source.data).hexdigest()
    result=normalize_cloud(source,ctx(),digest)
    assert not result.message.is_bigendian
    np.testing.assert_array_equal(np.frombuffer(result.message.data,'<f4').reshape(-1,3),[[1,2,3]])
    assert result.source_digest==digest and result.output_digest != digest
    assert source.is_bigendian
    assert result.context.time_basis=='receipt_estimated'
    assert result.context.ingress_monotonic_ns==100


def test_row_padding_removed_without_coordinate_change():
    source=cloud();source.width=1;source.height=2;source.point_step=12;source.row_step=16
    source.fields=source.fields[:3]
    source.data=struct.pack('<fffIfffI',1,2,3,42,4,5,6,99)
    result=normalize_cloud(source,ctx(),hashlib.sha256(source.data).hexdigest())
    np.testing.assert_array_equal(np.frombuffer(result.message.data,'<f4').reshape(-1,3),[[1,2,3],[4,5,6]])


def test_original_digest_mismatch_is_diagnostic_without_blocking_conversion():
    result=normalize_cloud(cloud(),ctx(),'0'*64)
    assert not result.source_digest_verified
    assert result.message.width == 1


@pytest.mark.parametrize('change',[lambda m:setattr(m,'row_step',1),lambda m:setattr(m,'data',m.data+b'x'),
    lambda m:setattr(m.fields[2],'offset',2)])
def test_invalid_layout_rejected(change):
    source=cloud();change(source)
    with pytest.raises(ValueError):normalize_cloud(source,ctx(),None)


def test_optional_bad_points_filtered():
    source=cloud(points=[(1,2,3),(float('nan'),2,3)])
    result=normalize_cloud(source,ctx(),None)
    assert result.message.width==1 and result.message.is_dense
    assert result.source_digest_verified is False

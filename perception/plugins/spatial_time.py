"""Immutable observation evidence; receipt time is not capture time."""
from dataclasses import asdict, dataclass, fields, replace
from typing import Optional, Tuple
import uuid

CLOCK_FIELDS = ('source_clock', 'mapping_offset_ns', 'clock_mapping',
                'clock_mapping_verified', 'clock_mapping_uncertainty_ns',
                'source_id', 'source_epoch')


def compatible_sources(first, second):
    """Retain compatibility API without treating source claims as policy."""
    return None  # Producer descriptions do not gate shared numeric input.


def _integer(value, name, minimum=None):
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(name + ' must be an integer >= ' + str(minimum))


@dataclass(frozen=True)
class FrameContext:
    observation_id: str
    observation_stamp_ns: int
    normalized_clock_id: str
    time_basis: str
    ingress_realtime_ns: int
    ingress_monotonic_ns: int
    ingress_clock_id: str
    source_id: Optional[str] = None
    source_epoch: Optional[str] = None
    source_capture_id: Optional[str] = None
    original_stamp: Optional[int] = None
    original_unit: Optional[str] = None
    original_clock_id: Optional[str] = None
    mapping_source: Optional[str] = None
    mapping_version: Optional[str] = None
    mapping_offset_ns: Optional[int] = None
    mapping_verified: bool = False
    uncertainty_ns: Optional[int] = None
    geometry_version: Optional[str] = None
    parent_observation_ids: Tuple[str, ...] = ()
    transformations: Tuple[str, ...] = ()

    def __post_init__(self):
        for name in ('observation_stamp_ns','ingress_realtime_ns','ingress_monotonic_ns'):
            _integer(getattr(self,name),name)
        for name in ('observation_id','normalized_clock_id','ingress_clock_id','time_basis'):
            if not isinstance(getattr(self,name),str):
                raise ValueError(name+' must be a string')
        for name in ('parent_observation_ids','transformations'):
            value=getattr(self,name)
            if not isinstance(value,tuple) or any(not isinstance(v,str) for v in value):
                raise ValueError(name+' must contain strings')

    def to_dict(self):
        return dict(schema='motus.observation/1', **asdict(self))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.get('schema') != 'motus.observation/1':
            raise ValueError('unsupported observation schema')
        data = {k:v for k,v in value.items() if k in {f.name for f in fields(cls)}}
        for name in ('observation_id','normalized_clock_id','ingress_clock_id'):
            data.setdefault(name,'')
        data.setdefault('time_basis','source_capture')
        for name in ('ingress_realtime_ns','ingress_monotonic_ns'):
            data.setdefault(name,0)
        for name in ('parent_observation_ids', 'transformations'):
            if name in data and isinstance(data[name], list):
                data[name] = tuple(data[name])
        try:
            return cls(**data)
        except TypeError as error:
            raise ValueError('incomplete observation context') from error


class TimePolicy:
    def __init__(self, ingress_clock_id, *, max_age_ns=1_000_000_000, max_future_ns=50_000_000):
        if not isinstance(ingress_clock_id, str) or not ingress_clock_id:
            raise ValueError('ingress clock identity required')
        _integer(max_age_ns, 'max_age_ns', 1)
        _integer(max_future_ns, 'max_future_ns')
        self.clock_id, self.max_age_ns, self.max_future_ns = ingress_clock_id, max_age_ns, max_future_ns

    def select(self, source, ingress_realtime_ns, ingress_monotonic_ns):
        stamp = source.get('stamp_ns')
        if stamp is not None:
            _integer(stamp, 'capture stamp')
        time_offset = source.get('time_offset_ns', 0)
        _integer(time_offset, 'time offset')
        offset,uncertainty=source.get('mapping_offset_ns'),source.get('clock_mapping_uncertainty_ns')
        verified=source.get('clock_mapping_verified',False)
        capture=stamp is not None
        return FrameContext(
            observation_id=uuid.uuid4().hex, source_id=source.get('source_id'),
            source_epoch=source.get('source_epoch'), source_capture_id=source.get('source_capture_id'),
            observation_stamp_ns=stamp+time_offset if capture else ingress_realtime_ns,
            normalized_clock_id=(source.get('clock_id') or self.clock_id) if capture else self.clock_id,
            time_basis=source.get('time_basis') or ('source_capture' if capture else 'receipt_estimated'),
            ingress_realtime_ns=ingress_realtime_ns, ingress_monotonic_ns=ingress_monotonic_ns,
            ingress_clock_id=self.clock_id, original_stamp=source.get('source_stamp', stamp),
            original_unit=source.get('source_unit', 'ns') if stamp is not None else None,
            original_clock_id=source.get('source_clock') or source.get('clock_id'),
            mapping_source=source.get('clock_mapping') if capture else 'perception_ingress',
            mapping_version=source.get('mapping_version'), mapping_offset_ns=offset if capture else None,
            mapping_verified=verified if capture else False, uncertainty_ns=uncertainty if capture else None,
            geometry_version=source.get('geometry_version'))


def receipt_association(context):
    """Derive arrival alignment while retaining the original capture evidence."""
    if context.time_basis == 'receipt_estimated':
        return context
    return replace(context, observation_id=uuid.uuid4().hex,
        observation_stamp_ns=context.ingress_realtime_ns,
        normalized_clock_id=context.ingress_clock_id, time_basis='receipt_estimated',
        mapping_source='perception_ingress', mapping_offset_ns=None,
        mapping_verified=False, uncertainty_ns=None,
        parent_observation_ids=(context.observation_id,),
        transformations=context.transformations+('arrival_alignment',))

"""Immutable observation evidence; receipt time is not capture time."""
from dataclasses import asdict, dataclass, fields, replace
from typing import Optional, Tuple
import uuid

CLOCK_FIELDS = ('source_clock', 'mapping_offset_ns', 'clock_mapping',
                'clock_mapping_verified', 'clock_mapping_uncertainty_ns',
                'source_id', 'source_epoch')


def compatible_sources(first, second):
    """Missing optional evidence is unknown; contradictory claims are errors."""
    for key in CLOCK_FIELDS:
        a, b = first.get(key), second.get(key)
        if key in first and key in second and a != b:
            raise ValueError('source clock declarations conflict: ' + key)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
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
        for name in ('observation_stamp_ns', 'ingress_realtime_ns'):
            _integer(getattr(self, name), name, 1)
        _integer(self.ingress_monotonic_ns, 'ingress_monotonic_ns')
        for name in ('observation_id', 'normalized_clock_id', 'ingress_clock_id'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(name + ' is required')
        for name in ('source_id', 'source_epoch', 'source_capture_id', 'original_unit',
                     'original_clock_id', 'mapping_source', 'mapping_version', 'geometry_version'):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError('invalid ' + name)
        if self.original_stamp is not None:
            _integer(self.original_stamp, 'original_stamp')
            if self.original_unit not in ('ns', 'us', 'ms'):
                raise ValueError('original timestamp unit is required')
        if self.mapping_offset_ns is not None and type(self.mapping_offset_ns) is not int:
            raise ValueError('mapping offset must be integer nanoseconds')
        if self.uncertainty_ns is not None:
            _integer(self.uncertainty_ns, 'uncertainty_ns')
        if type(self.mapping_verified) is not bool:
            raise ValueError('mapping_verified must be boolean')
        if self.time_basis not in ('source_capture', 'receipt_estimated'):
            raise ValueError('unsupported observation time basis')
        if self.time_basis == 'receipt_estimated' and (
                self.observation_stamp_ns != self.ingress_realtime_ns or
                self.normalized_clock_id != self.ingress_clock_id or
                self.mapping_verified or self.uncertainty_ns is not None):
            raise ValueError('receipt cannot claim capture time or verified acquisition accuracy')
        for name in ('parent_observation_ids', 'transformations'):
            value = getattr(self, name)
            if not isinstance(value, tuple) or any(not isinstance(v, str) or not v for v in value):
                raise ValueError(name + ' must be an immutable tuple of strings')

    def to_dict(self):
        return dict(schema='motus.observation/1', **asdict(self))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.get('schema') != 'motus.observation/1':
            raise ValueError('unsupported observation schema')
        data = {k: v for k, v in value.items() if k != 'schema'}
        if set(data) - {f.name for f in fields(cls)}:
            raise ValueError('unknown observation fields')
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
        offset, uncertainty = source.get('mapping_offset_ns'), source.get('clock_mapping_uncertainty_ns')
        if offset is not None and type(offset) is not int:
            raise ValueError('invalid declared mapping offset')
        if uncertainty is not None:
            _integer(uncertainty, 'declared clock uncertainty')
        verified = source.get('clock_mapping_verified', False)
        if type(verified) is not bool:
            raise ValueError('invalid declared clock verification')
        capture = bool(stamp and source.get('clock_id') and
                       source.get('timestamp_semantics') == 'source_capture_mapped')
        if capture and not -self.max_future_ns <= ingress_realtime_ns-stamp <= self.max_age_ns:
            raise ValueError('stale or future capture time; receipt fallback is forbidden')
        return FrameContext(
            observation_id=uuid.uuid4().hex, source_id=source.get('source_id'),
            source_epoch=source.get('source_epoch'), source_capture_id=source.get('source_capture_id'),
            observation_stamp_ns=stamp if capture else ingress_realtime_ns,
            normalized_clock_id=source['clock_id'] if capture else self.clock_id,
            time_basis='source_capture' if capture else 'receipt_estimated',
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

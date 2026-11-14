"""Frozen waiting budgets; never a permission to extend inertial prediction."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class PauseContext:
    started_monotonic_ns: int
    deadline_monotonic_ns: int
    budget_ns: int

    def __post_init__(self):
        if (any(type(v) is not int for v in (self.started_monotonic_ns,self.deadline_monotonic_ns,self.budget_ns)) or
                self.started_monotonic_ns < 0 or not 1_000_000_000 <= self.budget_ns <= 3_000_000_000 or
                self.deadline_monotonic_ns != self.started_monotonic_ns+self.budget_ns):
            raise ValueError('invalid bounded pause')

    def expired(self, now_monotonic_ns):
        return now_monotonic_ns > self.deadline_monotonic_ns

    def to_dict(self, *, anchor_stamp_ns, ingress_clock_id):
        return dict(schema='motus.pause/2', **asdict(self), anchor_stamp_ns=anchor_stamp_ns,
                    ingress_clock_id=ingress_clock_id, mapping_allowed=False)

    @classmethod
    def from_dict(cls, value):
        if (not isinstance(value,dict) or value.get('schema') != 'motus.pause/2' or
                value.get('mapping_allowed') is not False or type(value.get('anchor_stamp_ns')) is not int or
                value['anchor_stamp_ns'] <= 0 or not isinstance(value.get('ingress_clock_id'),str) or
                not value['ingress_clock_id']):
            raise ValueError('invalid pause declaration')
        try:return cls(*(value[k] for k in ('started_monotonic_ns','deadline_monotonic_ns','budget_ns')))
        except (KeyError,TypeError) as error:raise ValueError('incomplete pause') from error


class RecoveryPolicy:
    def __init__(self, version=2):
        self.version=version

    def begin(self, input_period_ns, processing_p95_ns, now_monotonic_ns):
        values=(input_period_ns,processing_p95_ns)
        if any(v is not None and (type(v) is not int or v < 0) for v in values):
            raise ValueError('invalid recovery timing evidence')
        budget=1_000_000_000
        if self.version == 2 and all(v is not None for v in values):
            budget=max(budget,min(3_000_000_000,3*input_period_ns+2*processing_p95_ns))
        return PauseContext(now_monotonic_ns,now_monotonic_ns+budget,budget)

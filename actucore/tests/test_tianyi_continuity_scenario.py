import copy
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('scenario',Path(__file__).parents[2]/'deploy/tianyi_continuity_scenario.py')
scenario=importlib.util.module_from_spec(spec);spec.loader.exec_module(scenario)


def test_supplement_preserves_source_and_labels_fresh_ordered_regrip_inputs():
    frame={'sequence':83,'deadman':True,'controllers':{'left':{'buttons':[0.,1.]},'right':{'buttons':[0.,1.]}}}
    source=[{'frame':frame}];points=[{'frame':frame} for _ in range(11)]
    before=copy.deepcopy((source,points))
    rows,phases=scenario.make_rows(source,points)
    assert (source,points)==before
    assert len(rows)==1450 and rows[-1]['received_ns']<30_000_000_000
    assert [r['frame']['sequence'] for r in rows]==list(range(1,1451))
    assert all(b['received_ns']-a['received_ns']==20_000_000 for a,b in zip(rows,rows[1:]))
    assert [p['phase'] for p in phases]==['release','baseline','sustained','failure_input','recovered','release_again','regrip_baseline','regrip_motion']
    for phase in phases:
        part=rows[phase['first_sequence']-1:phase['last_sequence']]
        held=not phase['phase'].startswith('release')
        assert all(r['frame']['deadman']==held for r in part)
        assert all(r['frame']['controllers']['left']['buttons'][1]==float(held) for r in part)
    assert rows[-1]['frame']['clutch_sequence']==2

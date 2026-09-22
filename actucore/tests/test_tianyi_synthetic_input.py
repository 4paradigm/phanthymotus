import runpy
import json
from pathlib import Path
import numpy as np
import pytest
from test_tianyi_arm_root import profile
from test_teleop import frame


def test_synthetic_input_is_reachable_and_separate_from_sensor_feedback(tmp_path,monkeypatch):
    root=Path(__file__).parents[2]
    monkeypatch.syspath_prepend(str(root/'actucore'))
    module=runpy.run_path(str(root/'deploy/test_tianyi_synthetic_input.py'))
    path,config,_=profile(tmp_path)
    config['controller_to_palm']={s:{'position':[.01,.02,.03],
        'orientation':[0.,0.,0.,1.]} for s in ('left','right')}
    path.write_text(json.dumps(config))
    solver=module['TianyiIK'](path)
    rows=module['generate'](solver,np.zeros(14),frame(1,True,1))
    assert len(rows)==501 and rows[-1]['received_ns']==10_000_000_000
    assert all(r['source']=='synthetic_fk_reference' and 'driver' not in r for r in rows)
    assert not rows[0]['frame']['deadman'] and not rows[-1]['frame']['deadman']
    assert rows[25]['frame']['deadman'] and rows[474]['frame']['deadman']
    q=np.asarray([r['reference_q'] for r in rows])
    assert np.max(np.abs(q[-1]))<1e-12
    assert np.ptp(q[:,0])==pytest.approx(.08) and np.ptp(q[:,7])==pytest.approx(.08)
    assert np.max(np.abs(np.diff(q,axis=0))/.02)<.027

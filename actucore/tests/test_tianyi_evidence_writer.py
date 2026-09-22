"""Slow or failed evidence storage must not silently alter replay input timing."""
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'plugins'))
from teleop.acceptance import _EvidenceWriter


def test_blocked_disk_does_not_block_feeder_or_context_exit_and_preserves_order():
    entered=threading.Event();release=threading.Event();lines=[]
    class File:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def write(self,text):
            entered.set()
            assert release.wait(2)
            lines.append(text)
    path=SimpleNamespace(open=lambda *args,**kwargs:File())
    writer=_EvidenceWriter(path,capacity=2)
    try:
        with writer:
            writer.write('first\n')
            assert entered.wait(1)
            writer.write('second\n');writer.write('third\n')
            with pytest.raises(ValueError,match='evidence_backlog'):writer.write('overflow\n')
        # A caller can stop hardware here before waiting for the blocked writer.
        assert writer.thread.is_alive() and not lines
    finally:
        release.set();writer.finish()
    assert lines==['first\n','second\n','third\n']
    with pytest.raises(ValueError,match='evidence_writer_closed'):writer.write('late\n')


def test_write_failure_is_not_reported_as_successful_flush():
    class File:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def write(self,text):raise OSError('disk full')
    writer=_EvidenceWriter(SimpleNamespace(open=lambda *a,**kw:File()))
    with writer:writer.write('evidence\n')
    with pytest.raises(ValueError,match='evidence_write_failed'):writer.finish()


def test_real_file_is_exclusive_and_all_rows_are_flushed(tmp_path):
    path=tmp_path/'evidence.jsonl'
    writer=_EvidenceWriter(path)
    with writer:
        for i in range(100):writer.write(str(i)+'\n')
    writer.finish()
    assert path.read_text().splitlines()==[str(i) for i in range(100)]
    with pytest.raises(FileExistsError):_EvidenceWriter(path)

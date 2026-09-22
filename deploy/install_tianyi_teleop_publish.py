#!/usr/bin/python3 -I
"""One-time USER sudo install; never run by an unattended build step."""
import argparse
import hashlib
import os
from pathlib import Path
import pwd
import subprocess
import yaml
import json


def install(source, expected, *, prefix=Path('/'), validate=True):
    raw=source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('reviewed_helper_hash_changed')
    if not raw.startswith(b'#!/usr/bin/python3 -I\n'):raise ValueError('isolated_python_required')
    executable=prefix/'usr/local/sbin/tianyi-teleop-publish'
    state=prefix/'var/lib/phanthy-teleop-publish'
    sudoers=prefix/'etc/sudoers.d/phanthy-teleop-publish'
    compose=prefix/'opt/phanthy-motus/docker-compose.yml'
    if executable.exists() or sudoers.exists() or state.exists():raise ValueError('existing_install_requires_review')
    config=yaml.safe_load(compose.read_text())
    for key in ('actucore-teleop','tianyi2'):config['services'][key].pop('image')
    executable.parent.mkdir(parents=True,exist_ok=True)
    sudoers.parent.mkdir(parents=True,exist_ok=True)
    state.mkdir(mode=0o700)
    executable.write_bytes(raw);executable.chmod(0o755)
    (state/'baseline.json').write_text(json.dumps(config,indent=2)+'\n')
    rule='nvidia ALL=(root) NOPASSWD: /usr/local/sbin/tianyi-teleop-publish *\n'
    candidate=state/'sudoers.candidate';candidate.write_text(rule);candidate.chmod(0o440)
    if validate:subprocess.run(['/usr/sbin/visudo','-cf',str(candidate)],check=True)
    sudoers.write_text(rule);sudoers.chmod(0o440)
    (state/'installation.json').write_text(json.dumps({'helper_sha256':expected,'user':'nvidia','actions':['check','apply','rollback']})+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--helper-sha256',required=True);a=p.parse_args()
    if os.geteuid()!=0 or pwd.getpwnam('nvidia').pw_uid!=1000:raise ValueError('expected_root_install_for_nvidia')
    install(Path(__file__).with_name('tianyi_teleop_publish.py'),a.helper_sha256)
    print('INSTALL PASS: fixed ActuCore/Driver publisher only; no services changed; no motion')

if __name__=='__main__':main()

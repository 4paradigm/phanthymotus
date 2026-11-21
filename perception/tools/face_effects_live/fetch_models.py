"""Explicit opt-in downloads. Runtime never fetches models or sends frames."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--models',nargs='+',choices=['yunet','osf0','osf3','mediapipe'],default=['yunet','osf0'])
    args=parser.parse_args()
    manifest=Path(__file__).resolve().parents[2]/'plugins/face_effects_live/models.json'
    models=json.loads(manifest.read_text())['models']
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    for name in args.models:
        spec=models[name]; target=root/spec['file']
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest()!=spec['sha256']:
                raise RuntimeError(f'{name}: existing model has unexpected digest; refusing overwrite')
            print(name,'already verified');continue
        temporary=None
        try:
            with urllib.request.urlopen(spec['url'],timeout=60) as response, tempfile.NamedTemporaryFile(dir=root,delete=False) as out:
                temporary=Path(out.name)
                data=response.read(spec['bytes']+1)
                if len(data)!=spec['bytes'] or hashlib.sha256(data).hexdigest()!=spec['sha256']:
                    raise RuntimeError(f'{name}: download hash/size mismatch')
                out.write(data)
            # Hard-link atomically without overwriting a concurrently created file.
            import os
            os.link(temporary,target)
            print(name,spec['license'],spec['sha256'])
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


if __name__=='__main__':main()

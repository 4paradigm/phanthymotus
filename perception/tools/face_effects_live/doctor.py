import argparse
import json
import platform
from pathlib import Path
from plugins.face_effects_live.backends import capabilities, model_path
from plugins.face_effects_live.engine import Engine


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--probe',action='store_true',help='initialize models/renderer; never open camera')
    parser.add_argument('--frame',help='optional local image for three real inference frames; no camera')
    args=parser.parse_args()
    report={'platform':platform.platform(),'architecture':platform.machine(),'capabilities':capabilities()}
    if args.config:
        cfg=json.loads(Path(args.config).read_text())
        report['models']={}
        for name in ('detector','landmark','face_landmarker'):
            path=cfg.get(name+'_model')
            if path:
                try:
                    model_path(path,cfg.get(name+'_sha256'))
                    report['models'][name]={'valid_file':True}
                except ValueError as exc:report['models'][name]={'valid_file':False,'error':str(exc)}
        if args.probe:
            engine=None
            try:
                engine=Engine(cfg)
                report['probe']={'ok':True,'renderer':engine.renderer.name,
                                 'fallback':engine.render_fallback,
                                 'configured_providers':getattr(engine.landmarker or engine.detector,'providers',['opencv-cpu'])}
                if args.frame:
                    import cv2
                    import numpy as np
                    from plugins.face_effects_live.contracts import Frame
                    cv2.setNumThreads(2)
                    image=cv2.imread(args.frame)
                    if image is None:raise ValueError('invalid probe image')
                    for k in range(3):result=engine.process(Frame(image,k/30,k))
                    report['frame_probe']={'frames':3,'tracked':result.metadata['tracked'],
                        'size':result.metadata['size'],'effects':result.metadata['effects'],
                        'changed_values':int(np.count_nonzero(result.image!=image)),
                        'scope':'local image real inference, not platform integration'}
            except Exception as exc:
                report['probe']={'ok':False,'error':type(exc).__name__}
            finally:
                if engine:engine.close()
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

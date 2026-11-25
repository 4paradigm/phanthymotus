"""Package compact actual evidence; does not rerun or invent measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from score_natural import score
from score_motion import score as motion_score


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(1048576):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--platform',default='platform-clean-final')
    args=p.parse_args();root=args.evidence;out=args.output;out.mkdir(parents=True,exist_ok=True)
    annotations=out/'natural-annotations-reviewed.json';labels=json.loads(annotations.read_text())
    comparisons={};quality={};motion={};external={}
    for folder in sorted((root/'final-comparison').iterdir()):
        comparisons[folder.name]=json.loads((folder/'summary.json').read_text())
        quality[folder.name]=score(folder,labels);motion[folder.name]=motion_score(folder,labels)
        external[folder.name]={p.name:sha(p) for p in sorted(folder.glob('*.jsonl'))}
        (folder/'quality.json').write_text(json.dumps(quality[folder.name],indent=2)+'\n')
        (folder/'motion.json').write_text(json.dumps(motion[folder.name],indent=2)+'\n')
    for name,value in [('comparison',comparisons),('quality',quality),('motion',motion)]:
        (out/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    manifest=json.loads((root/'natural/manifest.json').read_text())
    sources={}
    for name in ('interview','greeting'):
        path=root/'videos'/(name+'.webm');spec=dict(manifest['sources'][name])
        spec.update(bytes=path.stat().st_size,sha256=sha(path),file=str(path),
                    transfer_status='complete transfer; selected windows decoded; full recording not decoded')
        html=root/(name+'-source.html')
        if html.exists():spec['license_record_html_sha256']=sha(html)
        sources[name]=spec
        manifest['sources'][name].update(bytes=spec['bytes'],file_sha256=spec['sha256'])
    manifest['limits']=[v for v in manifest['limits'] if 'partial download' not in v]
    manifest['limits'].append('complete source transfers now present; only selected windows fully decoded')
    (out/'natural-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    sources['mediapipe_model_card']={'url':'https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf',
                                   'sha256':sha(root/'mediapipe-model-card.pdf')}
    (out/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    platform=json.loads((root/args.platform/'result.json').read_text())
    (out/'platform.json').write_text(json.dumps(platform,indent=2)+'\n')
    extra=root/'platform-monotonic-60-final/result.json'
    if extra.exists():
        load_platform=json.loads(extra.read_text())
        load_platform['derived_performance_gates']={'post_pause_all_displayed_frames_under_250ms':
            load_platform['pause_recovery']['max_post_resume_age_ms']<=250,'posthoc_from_recorded_frame_tags':True}
        (out/'platform-load.json').write_text(json.dumps(load_platform,indent=2)+'\n')
    for source,target in [('ros-adapters.json','ros-adapters.json'),('natural-preview/index.json','preview.json'),
                          ('reviewed-overlays/validation.json','annotation-validation.json')]:
        if (root/source).exists():shutil.copyfile(root/source,out/target)
    load={p.stem:json.loads(p.read_text()) for p in sorted((root/'load-final').glob('*.json')) if p.name!='status.json'}
    state=json.loads((root/'load-final/status.json').read_text())
    assert state.get('complete') and len(load)==8 and all(j['exit_code']==0 for j in state['jobs'])
    (out/'load.json').write_text(json.dumps(load,indent=2)+'\n')
    (out/'load-commands.json').write_text(json.dumps(state,indent=2)+'\n')
    external['annotations_sha256']=sha(annotations)
    external['baseline_source_sha256']={str(p.relative_to(root/'baseline/face_effects_live')):sha(p)
        for p in sorted((root/'baseline/face_effects_live').rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    external['external_root']=str(root.resolve())
    (out/'artifact-index.json').write_text(json.dumps(external,indent=2)+'\n')
    print('packaged',len(comparisons),'model runs,',len(load),'load runs, actual platform pass:',platform['pass'])


if __name__=='__main__':main()

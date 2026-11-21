"""Extract fixed, lossless natural-video windows; never synthesizes head motion."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2


WINDOWS = [
    ("handshake", "interview", 74., 8., ["two_faces", "head_rotation"]),
    ("profiles", "interview", 120., 8., ["two_faces", "profile"]),
    ("room_transition", "interview", 178., 8., ["walking", "occlusion", "scene_cut"]),
    ("close_speaker", "interview", 246., 8., ["expressions", "head_motion"]),
    ("close_listener", "interview", 306., 8., ["expressions", "head_motion"]),
    ("greeting_cross", "greeting", 439., 8., ["multiple_people", "crossing", "occlusion"]),
    ("greeting_cluster", "greeting", 449., 8., ["multiple_people", "profile", "occlusion"]),
    ("outdoor_flash", "greeting", 176., 8., ["multiple_people", "lighting", "small_faces"]),
]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(1048576):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evidence',required=True)
    parser.add_argument('--output',help='NEW output directory; existing clips are never overwritten')
    args=parser.parse_args()
    root=Path(args.evidence).resolve()
    output=Path(args.output).resolve() if args.output else root/'natural'
    output.mkdir(parents=True,exist_ok=True)
    if any((output/(w[0]+'.mkv')).exists() for w in WINDOWS):
        raise FileExistsError('choose a new output directory; preserving existing clip evidence')
    result={'schema':'face_effects_natural_dataset/1','clips':[],
            'provenance':'real archival video; only temporal selection and spatial resize',
            'sources':{
                'interview':{'url':'https://commons.wikimedia.org/wiki/File:The_YouTube_Interview_with_President_Obama.webm',
                             'author':'The White House', 'license':'public domain, PD-USGov-POTUS',
                             'restriction':'only original White House interview after 74s; excludes third-party introduction'},
                'greeting':{'url':'https://commons.wikimedia.org/wiki/File:President_Obama_welcomes_President-Elect_Trump_to_the_White_House_(25-12322-F).webm',
                            'author':'The White House / Barack Obama Presidential Library',
                            'license':'public domain, US federal government audiovisual record'}},
            'limits':['selected clips are not demographically representative',
                      'lighting changes and motion severity must be visually audited, not inferred from tags',
                      'all selected windows are decoded completely; full original recording decode is not asserted']}
    for source in result['sources']:
        path=root/'videos'/f'{source}.webm'
        result['sources'][source].update(file_sha256=digest(path),bytes=path.stat().st_size)
    for name,source,start,seconds,tags in WINDOWS:
        path=root/'videos'/f'{source}.webm'
        cap=cv2.VideoCapture(str(path))
        fps=cap.get(cv2.CAP_PROP_FPS)
        stride=max(1,round(fps/30));out_fps=fps/stride
        first=round(start*fps);count=round(seconds*out_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES,first)
        target=output/(name+'.mkv')
        writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'FFV1'),out_fps,(1280,720))
        if not writer.isOpened():raise RuntimeError('FFV1 encoder unavailable')
        pixel_hash=hashlib.sha256()
        for k in range(count):
            for s in range(stride):
                ok,image=cap.read()
                if not ok:raise RuntimeError(f'incomplete natural window {name}, frame {k}')
                if s==0:chosen=image
            image=cv2.resize(chosen,(1280,720),interpolation=cv2.INTER_AREA)
            pixel_hash.update(image.tobytes());writer.write(image)
        writer.release();cap.release()
        result['clips'].append({'id':name,'source':source,'source_start_seconds':start,
                                'source_first_frame':first,'source_stride':stride,
                                'source_fps':fps,'fps':out_fps,'frames':count,'tags':tags,
                                'size':[1280,720],'file':target.name,'sha256':digest(target),
                                'decoded_pixels_sha256':pixel_hash.hexdigest()})
        print(name,count,flush=True)
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

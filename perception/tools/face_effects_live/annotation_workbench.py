"""Source-only local annotation review; no models, uploads or original edits.

Prepare a NEW directory of source frames, then serve it on loopback. Reviewers
export JSON through the browser. Existing source labels are never overwritten.
"""
import argparse
import copy
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import struct
from urllib.parse import unquote


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(labels_path,clips,original_frames,output):
    import cv2
    import numpy as np
    labels=json.loads(labels_path.read_text());output.mkdir(parents=True,exist_ok=False)
    needed={}
    for clip,frames in labels['clips'].items():
        for frame in frames:needed[(clip,frame['frame_id'])]=frame.get('image_sha256')
    burst=labels.get('nose_burst')
    if burst:
        for frame in burst['frame_ids']:needed.setdefault((burst['clip'],frame),None)
    hashes={}
    for (clip,index),expected in sorted(needed.items()):
        if Path(f'{clip}-{index}.png').name!=f'{clip}-{index}.png':raise ValueError('invalid clip filename')
        cap=cv2.VideoCapture(str(clips/(clip+'.mkv')))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,image=cap.read()
            if not ok:raise ValueError(f'cannot decode source {clip}/{index}')
        finally:cap.release()
        name=f'{clip}-{index}.png';target=output/name;original=original_frames/name
        if original.exists():
            if not np.array_equal(cv2.imread(str(original)),image):raise ValueError('source pixel mismatch')
            if expected and sha(original)!=expected:raise ValueError('source image hash mismatch')
            shutil.copyfile(original,target)
        else:
            if expected:raise ValueError('original annotation image required to preserve its SHA')
            if not cv2.imwrite(str(target),image):raise RuntimeError('PNG encoding failed')
        hashes[name]=sha(target)
    (output/'sources.json').write_text(json.dumps({'image_sha256':hashes,'labels_sha256':sha(labels_path)},indent=2)+'\n')
    print(json.dumps({'prepared_source_images':len(hashes),'output':str(output)}))


def create_server(labels_path,frames,port=0):
    labels=json.loads(labels_path.read_text());items=[]
    for clip,annotations in labels['clips'].items():
        for k,annotation in enumerate(annotations):
            items.append({'clip':clip,'frame_id':annotation['frame_id'],'kind':'snapshot','index':k})
    burst=labels.get('nose_burst')
    if burst:
        for k,index in enumerate(burst['frame_ids']):
            items.append({'clip':burst['clip'],'frame_id':index,'kind':'nose','index':k})
    files={}
    for item in items:
        name=f"{item['clip']}-{item['frame_id']}.png";path=frames/name
        if Path(name).name!=name or not path.resolve().is_relative_to(frames.resolve()):
            raise ValueError('frame path escapes prepared source directory')
        if not path.is_file():raise ValueError('prepare all source frames before serving')
        with path.open('rb') as stream:header=stream.read(24)
        if len(header)<24 or header[:8]!=b'\x89PNG\r\n\x1a\n' or list(struct.unpack('>II',header[16:24]))!=labels['coordinate_space']:
            raise ValueError('source dimensions do not match annotation coordinate space')
        item['file']=name;item['image_sha256']=sha(path);files[name]=path
    for item in items:
        if item['kind']=='snapshot':
            expected=labels['clips'][item['clip']][item['index']].get('image_sha256')
            if expected and item['image_sha256']!=expected:raise ValueError('snapshot SHA mismatch')
    payload=json.dumps({'labels':labels,'items':items,'input_labels_sha256':sha(labels_path)},allow_nan=False).encode()
    html=Path(__file__).with_name('annotation_workbench.html').read_bytes()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            if self.path=='/':body,kind=html,'text/html; charset=utf-8'
            elif self.path=='/api/dataset':body,kind=payload,'application/json'
            elif self.path.startswith('/frames/') and unquote(self.path[8:]) in files:
                body,kind=files[unquote(self.path[8:])].read_bytes(),'image/png'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' blob:; connect-src 'self'")
            self.end_headers();self.wfile.write(body)
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


def main():
    p=argparse.ArgumentParser();p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--frames',type=Path,required=True);p.add_argument('--prepare',type=Path)
    p.add_argument('--clips',type=Path);p.add_argument('--port',type=int,default=0)
    args=p.parse_args()
    if args.prepare:
        if args.clips is None:p.error('--clips is required for preparation')
        prepare(args.labels,args.clips,args.frames,args.prepare);return
    server=create_server(args.labels,args.frames,args.port)
    print(f'Local source-only review: http://127.0.0.1:{server.server_port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()

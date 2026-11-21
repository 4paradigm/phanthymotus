"""Local real Driver runtime -> DDS -> Perception -> Core -> native Canvas.

No simulated middleware or replaced lifespan. Only the camera is a recorded
public-domain file. All runtime state goes to a NEW output directory. No robot
SDK, motion tool, container management or shared service is used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

ROOT = Path(__file__).resolve().parents[3]


def request(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type':'application/json'})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=20) as res:
        return json.load(res)


def serve_child(args):
    import cv2
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    cv2.setNumThreads(2)
    rclpy.init()
    executor = MultiThreadedExecutor(num_threads=2)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_:stop.set())
    signal.signal(signal.SIGINT, lambda *_:stop.set())
    worker = threading.Thread(target=executor.spin, daemon=True)
    worker.start()
    if args.role == 'perception':
        sys.path.insert(0, str(ROOT/'perception'))
        import main
        main._bundle = main.PerceptionBundle({'plugins':{
            'face_effects_live':{'enabled':True,'detector_model':str(args.models/'yunet.onnx'),
                                 'landmark_model':str(args.models/'osf0.onnx'),**json.loads(args.extra_config)},
            'audio_mixer':{'enabled':False},'echo_cancellation':{'enabled':False}}}, executor)
        bundle = main._bundle
        server = main.ThreadingHTTPServer(('127.0.0.1',0), main.make_handler())
        close = bundle.shutdown
    else:
        sys.path.insert(0, str(args.driver_root))
        from common.vendor_runtime import DriverBundle, make_handler
        from http.server import ThreadingHTTPServer
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CompressedImage

        class RecordedCamera:
            def __init__(self):
                self.node=Node('recorded_camera_acceptance')
                self.pub=self.node.create_publisher(CompressedImage,args.topic,
                    QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT))
                self.event=threading.Event();self.thread=None;self.count=0
                self.state='idle';self.lock=threading.RLock()
                self.error='';self.video_open=False

            def get_tools(self):
                return [{'name':'recorded_camera','type':'sensor',
                    'description':'Public-domain recorded video replay, no hardware SDK',
                    'inputSchema':{'type':'object','properties':{'action':{'type':'string','enum':['info','start','stop']}}},
                    'topic_out':[{'topic':args.topic,'format':'image/jpeg'}]}]

            def dispatch(self,action,parameters):
                with self.lock:
                    if action=='start' and not (self.thread and self.thread.is_alive()):
                        self.event.clear();self.state='running'
                        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
                    elif action=='stop':self.stop()
                    return {'ok':True,'state':self.state,'frames':self.count,'error':self.error,
                            'video_open':self.video_open,'subscribers':self.pub.get_subscription_count(),
                            'provenance':'natural recorded public-domain video, repeated',
                            'topic_out':[{'topic':args.topic,'format':'image/jpeg'}]}

            def run(self):
                cap=cv2.VideoCapture(str(args.video));deadline=time.monotonic()
                try:
                    self.video_open=cap.isOpened()
                    if not self.video_open:raise RuntimeError('recorded file did not open')
                    while not self.event.is_set():
                        ok,image=cap.read()
                        if not ok:cap.set(cv2.CAP_PROP_POS_FRAMES,0);continue
                        stamp=time.time_ns() # decode return, NOT sensor exposure
                        msg=CompressedImage();msg.header.stamp.sec=stamp//10**9
                        msg.header.stamp.nanosec=stamp%10**9
                        msg.header.frame_id=f'recorded:{self.count}:{stamp}:{time.monotonic_ns()}'
                        valid,data=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])
                        if not valid:raise RuntimeError('encode failed')
                        msg.format='jpeg';msg.data=data.tobytes();self.pub.publish(msg);self.count+=1
                        deadline+=1/args.source_fps
                        self.event.wait(max(0,deadline-time.monotonic()))
                        if time.monotonic()-deadline>.2:deadline=time.monotonic()
                except Exception as exc:
                    self.state='error';self.error=repr(exc)
                    import traceback
                    traceback.print_exc()
                finally:cap.release()

            def stop(self):
                self.event.set()
                if self.thread:self.thread.join(3)
                self.state='idle'

        plugin=RecordedCamera();bundle=DriverBundle([plugin])
        server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(lambda:bundle,'Recorded Camera Acceptance','recorded-camera-acceptance'))
        def close():
            bundle.stop_all();plugin.node.destroy_node()
    serving=threading.Thread(target=server.serve_forever,daemon=True);serving.start()
    import importlib.util
    spec=importlib.util.spec_from_file_location('local_dds_audit',ROOT/'agent-core/src/dds_isolation.py')
    dds_audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(dds_audit)
    (args.output/(args.role+'-ready.json')).write_text(json.dumps({'url':f'http://127.0.0.1:{server.server_port}/mcp'}))
    try:
        while not stop.wait(.5):
            transport={'external_dds_binds':dds_audit._external_dds_binds()}
            if args.role=='perception':transport['sse_clients']=len(main._sse_clients)
            (args.output/(args.role+'-transport.json')).write_text(json.dumps(transport))
    finally:
        close();server.shutdown();server.server_close();executor.shutdown(timeout_sec=3)
        rclpy.shutdown();worker.join(3)


PROBE_JS = r"""(() => {
  const original=URL.createObjectURL.bind(URL), tags=new Map();
  window.frameEvidence=[];
  URL.createObjectURL=function(blob){const url=original(blob);
    tags.set(url,blob.arrayBuffer().then(buffer=>{
      const a=new Uint8Array(buffer);let p=2;
      while(p+4<a.length && a[p]===255){const marker=a[p+1],n=(a[p+2]<<8)|a[p+3];
        if(marker===218 || marker===217 || n<2)break;
        if(marker===239 && a[p+4]===70 && a[p+5]===69 && a[p+6]===76 && a[p+7]===49 && a[p+8]===0){
          return JSON.parse(new TextDecoder().decode(a.slice(p+9,p+2+n)));}p+=2+n;
      }return null;
    }));if(tags.size>200)tags.delete(tags.keys().next().value);return url;};
  document.addEventListener('load',async e=>{
    const el=e.target;if(el.tagName!=='IMG'||!el.closest('.renderer-camera'))return;
    const loaded=Date.now(),loadedPerf=performance.now(),tag=await tags.get(el.src);if(!tag)return;
    requestAnimationFrame(()=>{window.frameEvidence.push({tag,loaded_ms:loaded,loaded_perf_ms:loadedPerf,raf_ms:Date.now(),raf_perf_ms:performance.now(),width:el.naturalWidth,height:el.naturalHeight});
      if(window.frameEvidence.length>2000)window.frameEvidence.shift();});
  },true);
})();"""


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    # Bindable random local DDS domain for this harness, never a Card default.
    import secrets
    for _ in range(100):
        domain=secrets.randbelow(90)+10;sockets=[]
        try:
            for port in range(7400+domain*250,7400+domain*250+250):
                s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.bind(('127.0.0.1',port));sockets.append(s)
            break
        except OSError:continue
        finally:
            for s in sockets:s.close()
    else:raise RuntimeError('no free isolated DDS domain')
    core_root=args.core_root or ROOT/'agent-core'
    profile=args.output/'dds-local.xml'
    profile.write_bytes((core_root/'deploy/dds-local.xml').read_bytes())
    transport_types=[element.text for element in ET.parse(profile).iter()
                     if element.tag.rsplit('}',1)[-1]=='type']
    # The repo's FastDDS loopback profile provides isolation. Its own notes
    # warn that ROS_LOCALHOST_ONLY=1 can break local discovery on this runtime.
    os.environ.update(ROS_DOMAIN_ID=str(domain),ROS_LOCALHOST_ONLY=str(args.localhost_only),
        FASTRTPS_DEFAULT_PROFILES_FILE=str(profile),ROS_LOG_DIR=str(args.output/'ros-logs'),
        RESOURCE_CENTER_URL='http://127.0.0.1:1',
        DOCKER_HOST='unix://'+str(args.output/'nonexistent-docker.sock'))
    os.environ.update(NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost')
    rmw=os.environ.get('RMW_IMPLEMENTATION','rmw_fastrtps_cpp')
    if rmw=='rmw_cyclonedds_cpp':
        if args.cyclone_profile is None:raise ValueError('explicit loopback CycloneDDS profile required')
        os.environ['CYCLONEDDS_URI']='file://'+str(args.cyclone_profile)
    children=[];logs=[];core=None;core_thread=None
    evidence={'schema':'face-effects-real-platform/1','domain':domain,'assertions':{},'errors':[],
        'card_extra_config':json.loads(args.extra_config),
        'core_lifespan':'unmodified full startup/shutdown',
        'driver_runtime':str(args.driver_root/'common/vendor_runtime.py'),
        'video_sha256':hashlib.sha256(args.video.read_bytes()).hexdigest(),
        'requested_source_fps':args.source_fps,
        'dds_policy':f'{rmw} explicit local profile {transport_types}; ROS_LOCALHOST_ONLY={args.localhost_only}',
        'dds_transport_types':transport_types,
        'dds_profile_sha256':hashlib.sha256(profile.read_bytes()).hexdigest(),
        'rmw':rmw,
        'canvas_view':args.view,
        'core_source':str(core_root),
        'core_transport_source_sha256':{str(p.relative_to(core_root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [core_root/'src/api/inspection.py',core_root/'web/js/detail-panel.js',
                      core_root/'web/js/renderers/camera.js',core_root/'web/js/monitor-dashboard.js',core_root/'web/js/dashboard.js']},
        'limits':['recorded camera, no physical device SDK','browser onload/rAF is not display scanout',
                  'local host Unix clock only; not a sensor latency measurement']}
    try:
        topic='/recorded_acceptance_'+str(os.getpid())+'/image'
        for role in ('driver','perception'):
            log=(args.output/(role+'.log')).open('w');logs.append(log)
            command=[sys.executable,str(Path(__file__).resolve()),'--role',role,'--output',str(args.output),
                     '--models',str(args.models),'--video',str(args.video),'--driver-root',str(args.driver_root),
                     '--topic',topic,'--source-fps',str(args.source_fps),'--extra-config',args.extra_config]
            children.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT))
        deadline=time.monotonic()+20
        while not all((args.output/(r+'-ready.json')).exists() for r in ('driver','perception')):
            if time.monotonic()>deadline or any(c.poll() is not None for c in children):raise RuntimeError('child startup failed')
            time.sleep(.1)
        work=args.output/'core-runtime';work.mkdir();(work/'web').symlink_to(core_root/'web',target_is_directory=True)
        os.environ['DB_PATH']=str(work/'data.db');os.chdir(work)
        sys.path.insert(0,str(core_root/'src'))
        import config
        config.main['core']={**config.main.get('core',{}),'configured':True,'main_loop_enable':False,'auto_start':False}
        config.main['services']={**config.main.get('services',{}),'resource_center':{'url':'http://127.0.0.1:1'}}
        config.main['peer_settings']={**config.main.get('peer_settings',{}),'enabled':False}
        import start
        import uvicorn
        sock=socket.socket();sock.bind(('127.0.0.1',0));base=f'http://127.0.0.1:{sock.getsockname()[1]}'
        core=uvicorn.Server(uvicorn.Config(start.app,log_level='warning',lifespan='on'))
        core_thread=threading.Thread(target=lambda:core.run(sockets=[sock]),daemon=True);core_thread.start()
        deadline=time.monotonic()+30
        while not core.started:
            if time.monotonic()>deadline:raise RuntimeError('full Core startup timeout')
            time.sleep(.1)
        mids={}
        for role in ('driver','perception'):
            url=json.loads((args.output/(role+'-ready.json')).read_text())['url']
            mids[role]=request(base+'/api/mcp',{'name':role+' local recorded acceptance','url':url,'category':role})['data']['id']
            # This Core revision assigns IDs with second precision. Distinct
            # registrations must not share a second; keep its implementation
            # unchanged and record this existing platform limitation.
            time.sleep(1.05)
        assert len(set(mids.values()))==2
        evidence['mcp_ids']=mids
        deadline=time.monotonic()+20
        while True:
            listing=request(base+'/api/mcp')['data']
            if all(any(m.get('id')==mid and m.get('tools') for m in listing) for mid in mids.values()):break
            if time.monotonic()>deadline:raise RuntimeError('real MCP discovery timeout')
            time.sleep(.1)
        evidence['assertions']['mcp_discovery']=True
        deadline=time.monotonic()+8
        while True:
            transport=args.output/'perception-transport.json'
            if transport.exists():
                try:
                    if json.loads(transport.read_text())['sse_clients']>0:break
                except ValueError:pass # child may be replacing its tiny status file
            if time.monotonic()>deadline:raise RuntimeError('real Core SSE subscription missing')
            time.sleep(.1)
        evidence['assertions']['core_subscribed_actual_perception_sse']=True
        def call(action,**kwargs):
            result=request(base+'/api/mcp/'+mids['perception']+'/call',{'tool':'face_effects_live','arguments':{'action':action,'instance_id':'effects-test',**kwargs}})
            data=result.get('data');content=data.get('content') if isinstance(data,dict) else data
            if not isinstance(content,list):raise RuntimeError('unexpected Core MCP response: '+str(result)[:500])
            value=json.loads(next(c['text'] for c in content if c.get('type')=='text'))
            if not value.get('ok'):raise RuntimeError('Card rejected call: '+str(value))
            return value
        call('config',source_kind='ros_compressed',detector_model=str(args.models/'yunet.onnx'),landmark_model=str(args.models/'osf0.onnx'),max_faces=8)
        config.main['canvas_layout']={'cards':[
            {'id':'replay-test','mcpId':mids['driver'],'toolName':'recorded_camera','driverName':'recorded video','x':30,'y':30},
            {'id':'effects-test','mcpId':mids['perception'],'toolName':'face_effects_live','driverName':'perception','x':450,'y':30}],
            'connections':[{'id':'video-link','fromCardId':'replay-test','toCardId':'effects-test','fromPortIdx':0,'toPortIdx':0,'format':'image/jpeg'}],
            'execConnections':[]}
        services=config.main.get('services',{});services['resource_center']={'url':base};config.main['services']=services
        evidence['start_project']=request(base+'/api/config/start-project',{})
        evidence['initial_info']=call('info')
        assert evidence['initial_info']['state']=='running'
        assert evidence['initial_info']['topic_in'][0]['topic']==topic
        evidence['assertions']['native_graph_start_and_input_resolution']=True
        from playwright.sync_api import sync_playwright
        assets=args.assets
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--no-sandbox'],executable_path=args.browser)
            page=browser.new_page(viewport={'width':1600,'height':1100});page.add_init_script(PROBE_JS)
            evidence['bus_connections']=[]
            def connected(ws):
                from urllib.parse import urlsplit,parse_qs
                url=urlsplit(ws.url)
                if url.path.startswith('/ws/bus/'):
                    connection={'path':url.path,'fresh':parse_qs(url.query).get('fresh')==['1'],'binary_frames':0}
                    evidence['bus_connections'].append(connection)
                    def received(event):
                        if isinstance(event,bytes):connection['binary_frames']+=1
                    ws.on('framereceived',received)
            page.on('websocket',connected)
            evidence['browser_version']=browser.version
            def calibrate():
                t0=time.monotonic()*1000
                clock=page.evaluate('({wall_ms:Date.now(),perf_ms:performance.now()})')
                t1=time.monotonic()*1000
                return {**clock,'host_before_ms':t0,'host_after_ms':t1,
                        'offset_ms':(t0+t1)/2-clock['perf_ms'],'half_rtt_ms':(t1-t0)/2}
            page.on('pageerror',lambda e:evidence['errors'].append(str(e)))
            assets_map={'https://cdn.jsdelivr.net/npm/three@0.164/build/three.module.js':assets/'dashboard-cdn/three.module.js',
                        'https://cdn.jsdelivr.net/npm/three@0.164/examples/jsm/controls/OrbitControls.js':assets/'dashboard-cdn/OrbitControls.js'}
            def route(r):
                if r.request.url in assets_map:r.fulfill(path=assets_map[r.request.url],content_type='text/javascript',headers={'Access-Control-Allow-Origin':'*'})
                elif r.request.url.startswith((base,'blob:','data:')):r.continue_()
                else:r.abort()
            page.route('**/*',route);page.goto(base,wait_until='domcontentloaded')
            # Navigation creates a new performance.timeOrigin. Calibrate the
            # actual Canvas document, never the preceding about:blank page.
            evidence['clock_calibration_start']=calibrate()
            card=page.locator('.canvas-card').filter(has=page.locator('.canvas-card-tool',has_text='face_effects_live'))
            card.wait_for(state='visible',timeout=15000)
            if args.view=='monitor':page.locator('.mode-toggle-btn[data-mode="monitor"]').click()
            else:card.locator('.canvas-view-btn').click()
            try:
                page.wait_for_function('window.frameEvidence.length >= 30',timeout=20000)
            except Exception:
                evidence['failure_card_info']=call('info')
                evidence['failure_driver_info']=request(base+'/api/mcp/'+mids['driver']+'/call',{'tool':'recorded_camera','arguments':{'action':'info'}})
                evidence['failure_browser']=page.evaluate('({count:window.frameEvidence?.length,cameras:[...document.querySelectorAll(".renderer-camera img")].map(el=>({src:el.src,width:el.naturalWidth}))})')
                from api import inspection
                evidence['failure_last_frames']={k:len(v) for k,v in inspection._last_frame.items()}
                raise
            page.wait_for_timeout(8000)
            evidence['assertions']['native_canvas_decoded_real_dds_images']=True
            if args.pause_js_ms:
                before=time.monotonic()*1000
                evidence['browser_pause']=page.evaluate('duration => {const start=Date.now(),perf=performance.now();const end=perf+duration;while(performance.now()<end){};return {start_ms:start,end_ms:Date.now(),start_perf_ms:perf,end_perf_ms:performance.now()}}',args.pause_js_ms)
                evidence['browser_pause']['host_elapsed_ms']=time.monotonic()*1000-before
                page.wait_for_timeout(2000)
            evidence['selection']=call('select_effects',effects=['glasses'])
            page.wait_for_timeout(2000)
            assert call('info')['result']['effects']==['glasses']
            evidence['assertions']['live_effect_switch']=True
            page.screenshot(path=str(args.output/'canvas-real.png'),full_page=True)
            evidence['browser_frames']=page.evaluate('window.frameEvidence')
            evidence['clock_calibration_end']=calibrate()
            evidence['steady_info']=call('info')
            from api import inspection
            output_topic=evidence['steady_info']['topic_out'][0]['topic']
            evidence['actual_ws_queue_capacities']=[q.maxsize for q in inspection._topic_queues.get(output_topic,[])]
            import dds_isolation
            evidence['dds_binding_audit']={'core':dds_isolation._external_dds_binds()}
            for role in ('driver','perception'):
                evidence['dds_binding_audit'][role]=json.loads((args.output/(role+'-transport.json')).read_text())['external_dds_binds']
            evidence['assertions']['all_three_dds_processes_loopback_only']=not any(evidence['dds_binding_audit'].values())
            evidence['stop_project']=request(base+'/api/config/stop-project',{})
            deadline=time.monotonic()+15
            while call('info')['state']!='idle':
                if time.monotonic()>deadline:raise RuntimeError('stop did not release Card')
                page.wait_for_timeout(100)
            evidence['assertions']['native_graph_stop']=True
            page.wait_for_timeout(700)
            evidence['image_after_stop']=page.evaluate('topic => {const scope=[...document.querySelectorAll(".monitor-card")].find(el=>el.dataset.topic===topic)||document;const el=scope.querySelector(".renderer-camera img");return {present:!!el,has_src:!!el?.getAttribute("src")}}',output_topic)
            evidence['restart_project']=request(base+'/api/config/start-project',{})
            page.wait_for_timeout(4000)
            assert call('info')['state']=='running'
            evidence['assertions']['native_graph_restart']=True
            request(base+'/api/config/stop-project',{});page.wait_for_timeout(1500)
            browser.close()
        import numpy as np
        samples=evidence['browser_frames'];latency=[];raf=[];wall_latency=[]
        first_clock=evidence['clock_calibration_start'];last_clock=evidence['clock_calibration_end']
        offset=(first_clock['offset_ms']+last_clock['offset_ms'])/2
        drift=abs(first_clock['offset_ms']-last_clock['offset_ms'])
        evidence['clock_uncertainty_ms']=max(first_clock['half_rtt_ms'],last_clock['half_rtt_ms'])+drift/2
        evidence['clock_offset_drift_ms']=drift
        for sample in samples:
            source=sample['tag']['source_frame_id'].split(':')
            if len(source)==4 and source[0]=='recorded':
                capture=int(source[3])/1e6
                sample['capture_to_load_monotonic_ms']=sample['loaded_perf_ms']+offset-capture
                latency.append(sample['capture_to_load_monotonic_ms']);raf.append(sample['raf_perf_ms']+offset-capture)
                wall_latency.append(sample['loaded_ms']-int(source[2])/1e6)
        evidence['latency_ms']={name:dict(zip(('p50','p95','p99'),np.percentile(values,[50,95,99]).tolist()))
                                for name,values in [('decode_return_to_browser_load',latency),('decode_return_to_browser_raf',raf)]}
        evidence['assertions']['frame_provenance_reached_browser']=len(latency)>=30
        evidence['assertions']['monotonic_clock_calibration_stable']=drift<5 and evidence['clock_uncertainty_ms']<10
        evidence['wall_clock_latency_ms']=dict(zip(('p50','p95','p99'),np.percentile(wall_latency,[50,95,99]).tolist()))
        evidence['performance_gates']={'stopped_stream_image_cleared':evidence['image_after_stop']['present'] and not evidence['image_after_stop']['has_src']}
        first_source=samples[0]['tag']['source_frame_id'].split(':')
        last_source=samples[-1]['tag']['source_frame_id'].split(':')
        evidence['observed_source_fps']=(int(last_source[1])-int(first_source[1]))*1e9/(int(last_source[3])-int(first_source[3]))
        if evidence.get('browser_pause'):
            resumed=evidence['browser_pause']['end_perf_ms']
            after=[s for s in samples if s['loaded_perf_ms']>=resumed]
            fresh=[s['loaded_perf_ms']-resumed for s in after if s['capture_to_load_monotonic_ms']<250]
            evidence['pause_recovery']={'post_resume_samples':len(after),
                'first_under_250ms_age_after_resume_ms':min(fresh) if fresh else None,
                'max_post_resume_age_ms':max(s['capture_to_load_monotonic_ms'] for s in after)}
            evidence['performance_gates']['post_pause_all_displayed_frames_under_250ms']=evidence['pause_recovery']['max_post_resume_age_ms']<=250
        # pass is connectivity/control/measurement validity. Performance gates
        # are separate so an observed latency failure is never hidden by it.
        evidence['pass']=all(evidence['assertions'].values()) and not evidence['errors']
    except Exception as exc:
        evidence['pass']=False;evidence['failure']=repr(exc)
        if 'page' in locals():
            try:
                page.screenshot(path=str(args.output/'failure.png'),full_page=True)
                evidence['failure_browser']=page.evaluate('({count:window.frameEvidence?.length,cameras:[...document.querySelectorAll(".renderer-camera img")].map(el=>({src:el.src,width:el.naturalWidth}))})')
                evidence['failure_card_info']=call('info')
            except Exception:pass
        import traceback
        traceback.print_exc()
    finally:
        if core:
            core.should_exit=True
            if core_thread:core_thread.join(15)
            evidence['core_stopped']=not core_thread.is_alive()
        for child in children:
            child.terminate()
            try:child.wait(10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        for log in logs:log.close()
        (args.output/'result.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({k:v for k,v in evidence.items() if k not in ('browser_frames','steady_info','initial_info')}))
    return 0 if evidence['pass'] else 1


def main():
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['driver','perception','test'],default='test')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--models',type=Path,required=True)
    p.add_argument('--video',type=Path,required=True);p.add_argument('--driver-root',type=Path,required=True)
    p.add_argument('--assets',type=Path);p.add_argument('--topic')
    p.add_argument('--browser',help='existing compatible Chromium executable; never downloaded by harness')
    p.add_argument('--source-fps',type=float,default=30,help='accelerated replay is load-only, not quality data')
    p.add_argument('--extra-config',default='{}',help='explicit Card algorithm profile in all child processes')
    p.add_argument('--pause-js-ms',type=int,default=0,help='controlled local browser-consumer stall')
    p.add_argument('--core-root',type=Path,help='real Core checkout/snapshot for a fair transport comparison')
    p.add_argument('--view',choices=['detail','monitor'],default='detail')
    p.add_argument('--localhost-only',choices=[0,1],type=int,default=0,help='runtime-specific ROS setting; loopback XML always required')
    p.add_argument('--cyclone-profile',type=Path,help='explicit local-only profile when selecting installed CycloneDDS RMW')
    args=p.parse_args()
    for key in ('output','models','video','driver_root','assets','core_root','cyclone_profile'):
        value=getattr(args,key)
        if value is not None:setattr(args,key,value.resolve())
    if args.role!='test':serve_child(args);return 0
    return run(args)


if __name__=='__main__':sys.exit(main())

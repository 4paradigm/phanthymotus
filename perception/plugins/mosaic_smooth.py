"""Opt-in weak-privacy demonstration processor Card for PhanthyMotus.

This Card is deliberately distinct from privacy_view: a smoothly moving,
approximate mosaic is an image *effect*, not guaranteed anonymization. It
must not be described or deployed as a security/privacy boundary.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict, deque
from dataclasses import asdict

from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from plugins.mosaic_smooth_runtime import (
    DemoConfig, MosaicDemoError, SmoothMosaic, validate_model, MAX_JPEG_BYTES,
)
from utils.model_downloader import ensure_mosaic_smooth_model
from utils.model_progress import fetch_status
from utils.qos import CAMERA_QOS
from utils.ros_lifecycle import dispose_node


PREFIX = "mosaic_smooth_demo"
SCHEMA = "motus.mosaic_smooth_demo/1"
MAX_INSTANCES = 4
log = logging.getLogger(__name__)

CONFIG_SCHEMA = {
    "type":"object","additionalProperties":False,
    "properties":{
        "consent_acknowledged":{
            "type":"boolean","default":False,
            "title":"已征得所有出镜测试人员同意",
            "description":"实验演示，不保证匿名；必须明确同意后才能启动",
        },
        "output_width":{"type":"integer","minimum":320,"maximum":960,
                        "default":640,"title":"输出宽度（像素）"},
        "detect_every_frames":{"type":"integer","minimum":1,"maximum":12,
                               "default":4,"title":"每 N 帧检测一次"},
        "confidence_threshold":{"type":"number","minimum":0.1,"maximum":0.9,
                                "default":0.28,"title":"检测阈值"},
        "max_faces":{"type":"integer","minimum":1,"maximum":8,
                     "default":6,"title":"最大人脸数量"},
        "jpeg_quality":{"type":"integer","minimum":35,"maximum":95,
                        "default":72,"title":"JPEG 质量"},
        "pixelate_block":{"type":"integer","minimum":8,"maximum":64,
                          "default":24,"title":"人脸像素块大小"},
        "fallback_block":{"type":"integer","minimum":16,"maximum":96,
                          "default":40,"title":"未能定位时整帧粗马赛克块大小"},
    },
}
TOOLS = [{
    "name":PREFIX,
    "type":"processor",
    "multiInstance":True,
    "description":(
        "实验用流畅马赛克：按帧间隔 YuNet 检测+短时 LK 光流，"
        "定位不可信时整帧粗马赛克；不保证人脸匿名，必须经过出镜人员同意。"
        "与 privacy_view 隐私保护 Card 无关，不可作为隐私安全控制。"
    ),
    "inputSchema":{
        "type":"object",
        "required":["action"],
        "properties":{
            "action":{"type":"string","enum":["start","stop","info","config"]},
            "input_topic":{"type":"string","description":"ROS2 JPEG camera topic"},
        },
        "x-action-params":{
            "start":{"params":["input_topic"],"description":"开始本次流畅马赛克演示"},
            "stop":{"params":[],"description":"停止演示，仅清理当前实例"},
            "info":{"params":["input_topic"],"description":"查看状态、计数和免责声明"},
            "config":{"params":[],"description":"设置本次演示参数与知情同意标记"},
        },
    },
    "configSchema":CONFIG_SCHEMA,
    "topic_in":[{"format":"image/jpeg","desc":"Camera JPEG"}],
    "topic_out":[
        {"format":"image/jpeg","desc":"Weak-privacy mosaic demo JPEG"},
        {"format":"data/json","desc":"Demo state and warnings"},
    ],
}]


def validate_topic(topic: str) -> str:
    if (not isinstance(topic,str) or len(topic)>200
        or not re.fullmatch(
            r"/(?:[A-Za-z_][A-Za-z_0-9]*/)*[A-Za-z_][A-Za-z_0-9]*",topic,
        )):
        raise MosaicDemoError("invalid_input_topic")
    return topic


def output_topics(input_topic: str, key: str) -> list[dict]:
    prefix = input_topic+"/mosaic_smooth_demo/instance_"+hashlib.sha256(key.encode()).hexdigest()[:16]
    return [
        {"topic":prefix+"/image","format":"image/jpeg"},
        {"topic":prefix+"/status","format":"data/json"},
    ]


class _DemoNode(Node):
    def __init__(self, key, topic, cfg, model_path, finished, retry=False):
        suffix=hashlib.sha256(key.encode()).hexdigest()[:16]
        super().__init__("mosaic_smooth_demo_"+suffix)
        self.key=key
        self.topic=topic
        self.cfg=cfg
        self.model_path=model_path
        self._finished=finished
        self._topics=output_topics(topic,key)
        self._gate=threading.RLock()
        self._cv=threading.Condition(self._gate)
        self._stop=threading.Event()
        self._thread=None
        self._latest=None
        self._latest_stamp=0
        self._last_stamp=0
        self._last_output_stamp=0
        self._last_output_time=0.0
        self._output_times=deque(maxlen=90)
        self._state="idle"
        self._reason="not_started"
        self._model_fetch = {
            "stage": "retrying" if retry else "pending",
            "retries": int(retry),
            "percent": None,
            "bytes_done": 0,
            "bytes_total": None,
        }
        self._received=0
        self._processed=0
        self._published=0
        self._rejected=0
        self._errors=0
        self._fallback=0
        self._detect_calls=0
        self._faces=0
        self._pub=self.create_publisher(
            CompressedImage,self._topics[0]["topic"],CAMERA_QOS,
        )
        self._status_pub=self.create_publisher(
            String,self._topics[1]["topic"],CAMERA_QOS,
        )
        self._sub=self.create_subscription(
            CompressedImage,topic,self._on_frame,CAMERA_QOS,
        )
        self._timer=self.create_timer(0.5,self._publish_status)

    def snapshot(self):
        with self._gate:
            times=self._output_times
            now=time.monotonic()
            fps=(
                (len(times)-1)/(times[-1]-times[0])
                if len(times)>=2 and times[-1]>times[0]
                and now-times[-1]<1.0 else 0.0
            )
            return {
                "schema":SCHEMA,"instance_id":self.key,
                "state":self._state,"reason":self._reason,
                "model_fetch":dict(self._model_fetch),
                "privacy_guaranteed":False,
                "warnings":[
                    "DEMO_ONLY_NOT_ANONYMIZATION",
                    "FACE_MISSES_AND_RECOGNIZABLE_PEOPLE_POSSIBLE",
                    "COARSE_FALLBACK_IS_NOT_OPAQUE",
                ],
                "input_topic":self.topic,
                "topic_in":[{"topic":self.topic,"format":"image/jpeg"}],
                "topic_out":copy.deepcopy(self._topics),
                "config":asdict(self.cfg),
                "metrics":{
                    "received":self._received,"processed":self._processed,
                    "published":self._published,"rejected":self._rejected,
                    "errors":self._errors,"fallback_frames":self._fallback,
                    "detection_calls":self._detect_calls,"faces":self._faces,
                },
                "effective_output_fps":round(fps,2),
                "last_output_stamp_ns":self._last_output_stamp,
                "output_age_s":(max(0.0,now-self._last_output_time)
                                if self._last_output_time else None),
                "fresh":bool(self._last_output_time
                             and now-self._last_output_time<.6),
            }

    def _publish_status(self):
        with self._gate:
            if self._state=="stopping":return
            msg = String()
            msg.data = json.dumps(self.snapshot(), ensure_ascii=False)
            self._status_pub.publish(msg)

    def _on_frame(self,msg):
        now_ns=time.time_ns()
        try:
            stamp=msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec
            data=bytes(msg.data)
            if (msg.format.lower() not in ("jpeg","jpg")
                or not 4<=len(data)<=MAX_JPEG_BYTES
                or stamp<=0 or now_ns-stamp>1_000_000_000
                or stamp-now_ns>100_000_000):
                raise MosaicDemoError("invalid_camera_frame")
        except (ValueError,AttributeError,TypeError,MosaicDemoError):
            with self._gate:self._rejected+=1
            return
        with self._cv:
            if self._stop.is_set() or self._state not in ("loading","running"):
                return
            if stamp<=self._last_stamp:
                self._rejected+=1
                return
            self._last_stamp=stamp
            self._received+=1
            # Only retain the newest frame: no unbounded queue and no raw files.
            self._latest=(data,msg.header,stamp,time.monotonic())
            self._cv.notify()

    def start(self):
        with self._gate:
            if self._stop.is_set():
                raise MosaicDemoError("already_stopping")
            if self._thread is not None:
                return
            self._state="loading"
            self._reason=("model_retrying" if self._model_fetch["retries"]
                          else "initializing_model")
            self._thread=threading.Thread(
                target=self._run,daemon=True,name="mosaic_smooth_"+self.get_name(),
            )
            self._thread.start()

    def _model_stage(self, stage):
        with self._gate:
            if self._stop.is_set():
                return
            self._model_fetch["stage"] = stage
            if stage == "retrying":
                self._model_fetch["retries"] += 1
            if stage == "downloading":
                self._model_fetch.update(percent=0, bytes_done=0)
            self._reason = "model_" + stage

    def _fetch_model(self):
        def status(text):
            # Called only inside progress() with the gate held.
            self._reason = text

        render_progress, _ = fetch_status(status, "YuNet")

        def progress(pct, mb_done, mb_total):
            with self._gate:
                if self._stop.is_set():
                    return
                self._model_fetch.update(
                    percent=pct, bytes_done=round(mb_done * 1024 ** 2),
                    bytes_total=round(mb_total * 1024 ** 2),
                )
                render_progress(pct, mb_done, mb_total)

        return ensure_mosaic_smooth_model(progress_cb=progress, stage_cb=self._model_stage)

    def _run(self):
        engine = None
        failure_reason = "model_or_runtime_failed"
        try:
            model_path = self.model_path
            if model_path is None:
                failure_reason = "model_download_failed"
                model_path = self._fetch_model()
                failure_reason = "model_or_runtime_failed"
            else:
                self._model_stage("verifying_local")
            if self._stop.is_set():
                return
            engine=SmoothMosaic(model_path,self.cfg)
            if self.model_path is not None:
                self._model_stage("local_verified")
            with self._gate:
                if not self._stop.is_set():
                    self._state="running"
                    self._reason="waiting_for_camera"
            while not self._stop.is_set():
                with self._cv:
                    self._cv.wait_for(
                        lambda:self._stop.is_set() or self._latest is not None,
                        timeout=0.25,
                    )
                    if self._stop.is_set():break
                    pending=self._latest
                    self._latest=None
                if pending is None:continue
                data,header,stamp,received_at=pending
                if time.monotonic()-received_at>.5:continue
                try:
                    encoded,faces,fallback=engine.process(data)
                except (Exception,) as exc:
                    with self._gate:
                        self._errors+=1
                        self._reason="processing_failed:"+type(exc).__name__
                    continue
                with self._gate:
                    self._processed+=1
                    self._detect_calls=engine.detect_calls
                    self._fallback+=int(fallback)
                    self._faces=faces
                    if (self._stop.is_set()
                        or time.monotonic()-received_at>.6
                        or stamp<=self._last_output_stamp):
                        continue
                    result=CompressedImage()
                    result.header=header
                    result.format="jpeg"
                    result.data=encoded
                    self._pub.publish(result)
                    self._published+=1
                    self._last_output_stamp=stamp
                    self._last_output_time=time.monotonic()
                    self._output_times.append(self._last_output_time)
                    self._reason=("full_frame_coarse_demo" if fallback
                                  else "face_roi_demo")
                    self._publish_status()
        except Exception as exc:
            with self._gate:
                if not self._stop.is_set():
                    if self._model_fetch["stage"] == "verification_failed":
                        failure_reason = "model_verification_failed"
                    elif (self._model_fetch["stage"] == "verifying_local"
                          and isinstance(exc, MosaicDemoError) and str(exc).startswith("model_")):
                        self._model_fetch["stage"] = "verification_failed"
                        failure_reason = "model_verification_failed"
                    elif failure_reason == "model_download_failed":
                        self._model_fetch["stage"] = "download_failed"
                    self._state="error"
                    self._reason=f"{failure_reason}:{type(exc).__name__}"
                    self._errors+=1
            log.exception("[%s] %s", PREFIX, failure_reason)
            self._publish_status()
        finally:
            with self._gate:
                self._state="stopping" if self._stop.is_set() else "error"
                self._stop.set()
                self._latest=None
            try:
                if engine is not None:
                    engine.close()
            finally:
                self._finished(self)

    def request_stop(self):
        with self._cv:
            self._stop.set()
            self._state="stopping"
            self._reason="stopped"
            if self._thread is not None and self._model_fetch["stage"] in (
                    "pending", "retrying", "verifying_cache", "waiting_for_lock",
                    "downloading", "verifying_download", "verifying_local"):
                self._model_fetch["stage"] = "cancelled"
            self._latest=None
            self._cv.notify_all()

    def join(self):
        if self._thread is not None:
            self._thread.join(timeout=2)


class MosaicSmoothDemoPlugin:
    PREFIX=PREFIX

    def __init__(self,cfg,executor):
        if not isinstance(cfg,dict):
            raise MosaicDemoError("plugin_config_must_be_object")
        allowed=set(DemoConfig.__dataclass_fields__)|{"enabled","model_path"}
        unknown=set(cfg)-allowed
        if unknown:
            raise MosaicDemoError("unknown_config_field:"+sorted(unknown)[0])
        self._defaults=DemoConfig().updated(
            {k:v for k,v in cfg.items() if k in DemoConfig.__dataclass_fields__}
        )
        self._model_path = (
            cfg.get("model_path") or os.environ.get("MOSAIC_SMOOTH_YUNET_MODEL") or None
        )
        self._executor=executor
        self._lock=threading.RLock()
        self._nodes={}
        self._configs={}
        self._start_errors = OrderedDict()
        self._closed=False

    def get_tools(self):
        return copy.deepcopy(TOOLS)

    def _finished(self,node):
        dispose_node(self._executor,node,label=PREFIX)
        with self._lock:
            if self._nodes.get(node.key) is node:
                result = node.snapshot()
                if result["state"] == "error":
                    self._start_errors[node.key] = result
                    if len(self._start_errors) > MAX_INSTANCES:
                        self._start_errors.popitem(last=False)
                del self._nodes[node.key]

    def dispatch(self,name,args):
        if not isinstance(args,dict):return {"state":"error","error":"invalid_args"}
        action=args.get("action",name)
        key=args.get("instance_id") or "_default"
        if not isinstance(key,str) or len(key)>256:
            return {"state":"error","error":"invalid_instance_id"}
        try:
            if action=="info":
                with self._lock:
                    node=self._nodes.get(key)
                    cfg=self._configs.get(key,self._defaults)
                    error = self._start_errors.get(key)
                if node is not None:return node.snapshot()
                if error is not None:return copy.deepcopy(error)
                topic=args.get("input_topic")
                topic=validate_topic(topic) if topic is not None else None
                return {
                    "state":"idle","instance_id":key,
                    "privacy_guaranteed":False,
                    "warnings":["DEMO_ONLY_NOT_ANONYMIZATION"],
                    "config":asdict(cfg),
                    "topic_in":[{"topic":topic,"format":"image/jpeg"}]
                    if topic else [],
                    "topic_out":output_topics(topic,key) if topic else [],
                }
            if action=="config":
                values=args.get("config")
                if values is None:
                    values={k:v for k,v in args.items()
                            if k not in ("action","instance_id")}
                with self._lock:
                    if self._closed:raise MosaicDemoError("plugin_closed")
                    if key in self._nodes:raise MosaicDemoError("stop_before_config")
                    current=self._configs.get(key,self._defaults)
                    result=current.updated(values)
                    self._configs[key]=result
                    self._start_errors.pop(key, None)
                return {"state":"configured","config":asdict(result),
                        "privacy_guaranteed":False}
            if action=="start":
                topic=args.get("input_topic")
                topics=args.get("input_topics")
                if topics is not None:
                    if not isinstance(topics,list) or len(topics)!=1:
                        raise MosaicDemoError("exactly_one_camera_required")
                    if topic is not None and topic!=topics[0]:
                        raise MosaicDemoError("conflicting_input_topics")
                    topic=topics[0]
                topic=validate_topic(topic)
                with self._lock:
                    if self._closed:raise MosaicDemoError("plugin_closed")
                    cfg=self._configs.get(key,self._defaults)
                    if not cfg.consent_acknowledged:
                        raise MosaicDemoError("consent_acknowledgment_required")
                    node=self._nodes.get(key)
                    if node is not None and node.topic!=topic:
                        raise MosaicDemoError("stop_before_change_camera")
                    if node is None:
                        if len(self._nodes)>=MAX_INSTANCES:
                            raise MosaicDemoError("max_instances_exceeded")
                        if self._model_path is not None:
                            validate_model(self._model_path)
                        node=_DemoNode(
                            key,topic,cfg,self._model_path,self._finished,
                            retry=key in self._start_errors,
                        )
                        try:
                            self._executor.add_node(node)
                        except Exception:
                            node.destroy_node()
                            raise
                        self._start_errors.pop(key, None)
                        self._nodes[key]=node
                    node.start()
                return node.snapshot()
            if action=="stop":
                with self._lock:
                    nodes=[self._nodes[key]] if args.get("instance_id") and key in self._nodes else (
                        [] if args.get("instance_id") else list(self._nodes.values())
                    )
                    for node in nodes:node.request_stop()
                    if args.get("instance_id"):
                        self._configs.pop(key,None)
                        self._start_errors.pop(key, None)
                    else:
                        self._configs.clear()
                        self._start_errors.clear()
                for node in nodes:node.join()
                return {"state":"idle" if not any(
                    node._thread and node._thread.is_alive()
                    for node in nodes) else "stopping",
                        "image_publication_disabled":True}
            raise MosaicDemoError("unknown_action")
        except (MosaicDemoError,ValueError,TypeError) as exc:
            return {"state":"error","error":str(exc),
                    "privacy_guaranteed":False}

    def shutdown(self):
        with self._lock:
            self._closed=True
            nodes=list(self._nodes.values())
            for node in nodes:node.request_stop()
        for node in nodes:node.join()


__all__=["MosaicSmoothDemoPlugin","TOOLS","SCHEMA","validate_topic","output_topics"]

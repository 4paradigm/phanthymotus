"""Measured local scan motion for asynchronous RGB/scan projection, never map poses."""
from collections import OrderedDict, deque
import math
import threading

import numpy as np

from .pointcloud_slam_alignment import align_pose
from .pointcloud_slam_odom import PoseSample
from .pointcloud_slam_transforms import pose_matrix


class MotionPending(ValueError):
    """The admitted observation is waiting for measured inter-capture motion."""


class MotionTimeline:
    def __init__(self, capacity):
        self.capacity=capacity
        self.pending=OrderedDict()
        self.poses=deque(maxlen=capacity)
        self._invalid=deque(maxlen=capacity)
        self._seed=None
        self.lock=threading.RLock()
        self.accepted=self.rejected=0

    def begin(self, stamp):
        with self.lock:
            self.pending.pop(stamp,None)
            self._invalid=deque((s for s in self._invalid if s!=stamp),maxlen=self.capacity)

    def invalidate(self, stamp):
        with self.lock:
            self.pending.pop(stamp,None)
            self._invalid.append(stamp)
            self.poses.clear()
            self._seed=None
            self.rejected+=1

    def observe(self, kind, message):
        stamp=message.header.stamp.sec*10**9+message.header.stamp.nanosec
        with self.lock:
            if stamp in self._invalid:return False
            pair=self.pending.setdefault(stamp,{})
            pair[kind]=message
            while len(self.pending)>self.capacity:
                old,_=self.pending.popitem(last=False)
                self.invalidate(old)
            info=pair.get('info')
            if info is not None and info.lost:
                self.invalidate(stamp)
                return True
            if set(pair)!={'pose','info'}:return False
            self.pending.pop(stamp,None)
            try:
                cov=np.asarray(info.covariance,float).reshape(6,6)
                pose=pair['pose'].pose.pose
                position=tuple(float(getattr(pose.position,k)) for k in ('x','y','z'))
                q=np.array([getattr(pose.orientation,k) for k in ('x','y','z','w')],float)
                norm=np.linalg.norm(q)
                if (not np.isfinite(cov).all() or np.any(np.diag(cov)<=0) or
                        not all(math.isfinite(x) for x in position) or not np.isfinite(q).all() or norm==0):
                    raise ValueError('invalid native motion result')
                sample=PoseSample(stamp//10**6,'local','body','local',position,tuple(q/norm),
                                  tuple(cov.ravel()),'local',None,None,stamp)
                if info.icp_correspondences<=0:
                    # Native initialization defines a gauge, not observed motion.
                    self.poses.clear()
                    self._seed=sample
                    return True
                if (not math.isfinite(info.icp_inliers_ratio) or info.icp_inliers_ratio<=0 or
                        not math.isfinite(info.icp_structural_complexity) or info.icp_structural_complexity<.02):
                    raise ValueError('native scan motion is unobservable')
                if self._seed is not None:
                    self.poses.append(self._seed)
                    self._seed=None
                self.poses.append(sample)
                self.accepted+=1
            except (ValueError,TypeError,AttributeError,OverflowError):
                self.invalidate(stamp)
            return True

    def relative(self, rgb_stamp, cloud_stamp):
        with self.lock:
            try:
                rgb=align_pose(self.poses,rgb_stamp,max_gap_ms=0)
                cloud=align_pose(self.poses,cloud_stamp,max_gap_ms=0)
                return np.linalg.inv(pose_matrix(cloud))@pose_matrix(rgb)
            except ValueError:pass
        raise MotionPending('measured motion does not cover both RGB and cloud captures')


def motion_parameters(settings, frame):
    """Native motion prediction initializes ICP; geometry must still validate it."""
    return dict(frame_id=frame,odom_frame_id=frame+'_odom',publish_tf=False,
        publish_null_when_lost=True,guess_frame_id='',min_update_rate=0.,qos=1,
        **{'Reg/Strategy':'1','Odom/Strategy':'0','Odom/ResetCountdown':'1',
           'Odom/GuessMotion':'true','OdomF2M/ScanMaxSize':'2000',
           'Icp/Strategy':'0','Icp/PointToPlane':'true',
           'Icp/PointToPlaneMinComplexity':'0.02','Icp/PointToPlaneLowComplexityStrategy':'0',
           'Icp/VoxelSize':str(settings['voxel_size_m'])})


class ScanMotion:
    """One supervised native local estimator, bounded input/results and no map."""
    def __init__(self, node, namespace, settings, directory, wake):
        self.node,self.namespace,self.settings=node,namespace,settings
        self.directory,self.wake=directory,wake
        self.timeline=MotionTimeline(max(8,settings['max_pending']*4))
        self._condition=threading.Condition()
        self._queue=deque(maxlen=settings['max_pending'])
        self._stop=threading.Event();self._result=threading.Event()
        self._process=self._thread=self._log=self._log_thread=self._inflight=None
        self._publisher=None
        self._subscriptions=[]
        self._generation=0
        self._current_namespace=namespace
        self.error=None;self.dropped=0

    def submit(self, stamp, points, matrix):
        # The bound is real downsampling, not scan_cloud_max_points metadata.
        points=points[::max(1,(len(points)+19999)//20000)]
        points=np.ascontiguousarray(points@matrix[:3,:3].T+matrix[:3,3],dtype='<f4')
        with self._condition:
            if self._stop.is_set() or self.error:return
            if len(self._queue)==self._queue.maxlen:self.dropped+=1
            self._queue.append((stamp,points))
            if self._thread is None:
                self._thread=threading.Thread(target=self._run,daemon=True,name='slam-local-scan-motion')
                self._thread.start()
            self._condition.notify()

    def _observe(self, kind, message, generation=None):
        stamp=message.header.stamp.sec*10**9+message.header.stamp.nanosec
        with self._condition:
            if (self._stop.is_set() or stamp!=self._inflight or self._result.is_set() or
                    generation is not None and generation!=self._generation):return
            if kind=='pose':
                q=message.pose.pose.orientation
                if (q.x,q.y,q.z,q.w)==(0.,0.,0.,0.):return  # Native explicit absent estimate.
            if self.timeline.observe(kind,message):
                self._result.set()
                self.wake.set()

    def _start_native(self):
        import json,os,subprocess,time
        from pathlib import Path
        from ament_index_python.packages import get_package_prefix
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import PointCloud2
        from rtabmap_msgs.msg import OdomInfo
        from rclpy.qos import QoSProfile,ReliabilityPolicy
        from rclpy.qos_event import PublisherEventCallbacks, SubscriptionEventCallbacks
        from utils.dds_transport import receive_environment
        executable=Path(get_package_prefix('rtabmap_odom'))/'lib/rtabmap_odom/icp_odometry'
        if not executable.is_file() or not os.access(executable,os.X_OK):
            raise RuntimeError('pinned native ICP odometry runtime unavailable')
        self.directory.mkdir(parents=True,exist_ok=True)
        with self._condition:
            self._generation+=1
            generation=self._generation
        namespace=self.namespace if generation==1 else self.namespace+'/run_'+str(generation)
        self._current_namespace=namespace
        frame=namespace.lstrip('/')+'/body'
        parameters=self.directory/'motion.params.json'
        parameters.write_text(json.dumps({'/**':{'ros__parameters':motion_parameters(self.settings,frame)}}))
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE)
        # Retry replaces these private handles while the executor is spinning.
        # Humble's default QoS event can hang while its wait set releases a
        # destroyed handle. The reliable profiles themselves stay in use.
        publisher=self._publisher=self.node.create_publisher(PointCloud2,namespace+'/scan_cloud',qos,
            event_callbacks=PublisherEventCallbacks(use_default_callbacks=False))
        for kind,cls,topic in [('pose',Odometry,'odom'),('info',OdomInfo,'odom_info_lite')]:
            self._subscriptions.append(self.node.create_subscription(cls,namespace+'/'+topic,
                lambda msg,k=kind,g=generation:self._observe(k,msg,g),qos,
                event_callbacks=SubscriptionEventCallbacks(use_default_callbacks=False)))
        self._log=(self.directory/'motion.log').open('w+b',buffering=0)
        (self.directory/'dds').mkdir(exist_ok=True)
        self._process=subprocess.Popen([str(executable),'--ros-args','--disable-external-lib-logs','--params-file',str(parameters),
            '-r','__ns:='+namespace,'-r','__node:=icp_odometry'],
            stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True,
            env=receive_environment(self.directory/'dds'))
        self._log_thread=threading.Thread(target=self._drain_log,daemon=True,name='slam-local-motion-log')
        self._log_thread.start()
        deadline=time.monotonic()+30.
        while not publisher.get_subscription_count() and not self._stop.wait(.02):
            if self._process.poll() is not None:raise RuntimeError('native scan motion exited during startup')
            if time.monotonic()>deadline:raise RuntimeError('native scan motion discovery timed out')
        return publisher,frame

    def _run(self):
        import time
        from sensor_msgs.msg import PointCloud2,PointField
        publisher=None
        try:
            if self._stop.is_set():return
            publisher,frame=self._start_native()
            while not self._stop.is_set():
                with self._condition:
                    self._condition.wait_for(lambda:self._queue or self._stop.is_set(),timeout=.25)
                    if self._stop.is_set():break
                    if not self._queue:
                        if publisher is not None and self._process.poll() is not None:raise RuntimeError('native scan motion exited')
                        continue
                    stamp,points=self._queue.popleft()
                if publisher is None:
                    if self._stop.is_set():break
                    publisher,frame=self._start_native()
                if self._stop.is_set():break
                message=PointCloud2()
                message.header.stamp.sec,message.header.stamp.nanosec=divmod(stamp,10**9)
                message.header.frame_id=frame
                message.height,message.width=1,len(points)
                message.fields=[PointField(name=n,offset=i*4,datatype=7,count=1) for i,n in enumerate(('x','y','z'))]
                message.point_step,message.row_step=12,len(points)*12
                message.is_bigendian,message.is_dense=False,True
                from array import array
                message.data=array('B',points.tobytes())
                with self._condition:
                    self.timeline.begin(stamp)
                    self._inflight=stamp;self._result.clear()
                publisher.publish(message)
                until=time.monotonic()+max(1.,self.settings['max_update_ms']/1000.)
                timed_out=False
                while not self._stop.is_set() and not self._result.wait(.02):
                    if time.monotonic()>until:
                        with self._condition:
                            if not self._result.is_set():
                                self._inflight=None
                                self.timeline.invalidate(stamp);self.wake.set()
                                timed_out=True
                        break
                    if self._process.poll() is not None:raise RuntimeError('native scan motion exited')
                with self._condition:self._inflight=None
                if timed_out:
                    self._terminate()
                    publisher=None
        except Exception as error:
            self.error=str(error)
            self.wake.set()
        finally:
            self._terminate()

    def _drain_log(self):
        try:
            while chunk:=self._process.stdout.read(65536):
                if self._log.tell()+len(chunk)>1024*1024:
                    self._log.seek(0);self._log.truncate()
                self._log.write(chunk)
        except OSError as error:
            self.error=str(error);self._stop.set();self.wake.set()

    def _terminate(self):
        import signal,subprocess
        with self._condition:
            self._inflight=None
            self._generation+=1  # Retire callbacks already queued by the old ROS handles.
        try:
            if self._process is not None:
                if self._process.poll() is None:
                    try:self._process.send_signal(signal.SIGINT)
                    except ProcessLookupError:pass
                    try:self._process.wait(timeout=5.)
                    except subprocess.TimeoutExpired:
                        self._process.kill();self._process.wait(timeout=2.)
                else:self._process.wait()
                if self._log_thread is not None:self._log_thread.join(timeout=2.)
                if self._process.stdout is not None:self._process.stdout.close()
        finally:
            if self._log is not None:self._log.close();self._log=None
            for subscription in self._subscriptions:self.node.destroy_subscription(subscription)
            self._subscriptions.clear()
            if self._publisher is not None:
                self.node.destroy_publisher(self._publisher);self._publisher=None

    def close(self):
        self._stop.set();self._result.set()
        with self._condition:
            self._queue.clear();self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=10.)
            if self._thread.is_alive():raise RuntimeError('native scan motion shutdown timed out')
        self._terminate()
        with self.timeline.lock:
            self.timeline.pending.clear();self.timeline.poses.clear();self.timeline._seed=None

    def status(self):
        with self._condition:
            queued=len(self._queue)
        return dict(state='unavailable' if self.error else 'estimating' if self._thread else 'waiting',
            reason=self.error,queued=queued,dropped=self.dropped,namespace=self._current_namespace,
            process_pid=self._process.pid if self._process is not None else None,
            stopped=self._stop.is_set(),worker_alive=bool(self._thread and self._thread.is_alive()),
            verified=self.timeline.accepted,rejected=self.timeline.rejected)

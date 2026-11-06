"""Bounded local graph batches with frozen, separately retained map history.

The overlap anchor maintains one source segment's map coordinates. No cross-batch
loop acceptance or global optimization of frozen history is claimed.
"""
import time
import numpy as np
from .pointcloud_slam_keyframes import Segment
from .pointcloud_slam_loop_search import ExperimentalLoopSearch
from .pointcloud_slam_map import rebuild_segment, map_from_odom
from .pointcloud_slam_stream import _matrix


class RollingGeometryHistory:
    def __init__(self, options, *, window_frames, window_points, camera=None, optical_from_child=None):
        if type(window_frames) is not int or window_frames < 2 or type(window_points) is not int or window_points <= 0:
            raise ValueError('bounded window must fit at least two keyframes')
        self.options, self.window_frames, self.window_points = dict(options), window_frames, window_points
        self.segment_id = None
        self.poses, self.frames = [], ()
        self.start, self.submaps = 0, 1
        self.camera, self.optical = camera, optical_from_child
        self.worker = self._worker()
        self.anchor = None
        self.frozen_edges = []
        self.frozen_attempts = self.frozen_skips = self.frozen_unindexed = 0

    def _worker(self):
        worker = ExperimentalLoopSearch(**self.options)
        if self.camera and 'max_forward_depth_score_m' in self.options['thresholds']:
            from .pointcloud_slam_revisit import RevisitSearch
            worker.revisit = RevisitSearch(self.camera, self.optical, self.options)
        return worker

    def update(self, segment, *, max_points):
        began = time.perf_counter()
        if (not segment.frames or (self.segment_id is not None and self.segment_id != segment.segment_id) or
                len(segment.frames) < len(self.frames) or
                any(a is not b for a,b in zip(segment.frames,self.frames))):
            raise ValueError('one append-only source segment required')
        self.segment_id = segment.segment_id
        descriptor_ms = graph_ms = 0.
        while True:
            end, count = self.start, 0
            while end < min(len(segment.frames),self.start+self.window_frames):
                size = len(segment.frames[end].points)
                if count + size > self.window_points: break
                count += size; end += 1
            if end == self.start or (end == self.start+1 and end < len(segment.frames)):
                raise ValueError('active point budget must fit overlap and next frame')
            part = Segment(segment.segment_id,segment.frames[self.start:end])
            raw = [_matrix(f.pose) for f in part.frames]
            graph = {'segment_id':segment.segment_id,'poses':raw,
                     'edges':[{'from':i-1,'to':i,'transform':np.linalg.inv(raw[i-1])@raw[i]}
                              for i in range(1,len(raw))]}
            loop = self.worker.update(part,graph,online=True,max_points=max_points,build_map=False)
            if self.anchor is None: self.anchor = raw[0].copy()
            align = self.anchor @ np.linalg.inv(loop['poses'][0])
            mapped = [align @ p for p in loop['poses']]
            self.poses[self.start:end] = mapped
            descriptor_ms += loop['timings_ms']['descriptor']
            graph_ms += loop['timings_ms']['registration_and_graph']
            edges = [{'source_stamp_ns': part.frames[c['source']].cloud_stamp_ns,
                      'target_stamp_ns': part.frames[c['target']].cloud_stamp_ns}
                     for c in loop['passed_candidates']]
            for edge, candidate in zip(edges, loop['passed_candidates']):
                if 'confirmation_stamps_ns' in candidate:
                    edge['target_from_source'] = candidate['target_from_source']
                    edge['confirmation_stamps_ns'] = candidate['confirmation_stamps_ns']
            if end == len(segment.frames): break
            self.frozen_edges.extend(edges)
            self.frozen_attempts += loop['candidate_attempts']
            self.frozen_skips += loop['descriptor_skips']
            self.frozen_unindexed += loop['descriptor_unindexed']
            self.start = end-1
            self.anchor = mapped[-1].copy()
            self.worker = self._worker()
            self.submaps += 1
        self.frames = segment.frames
        tick = time.perf_counter()
        points = rebuild_segment(segment,self.poses,max_points=max_points)
        return {'poses':list(self.poses),'after_map':points,
                'map_from_odom':map_from_odom(self.poses[-1],_matrix(segment.frames[-1].pose)),
                'passed_candidates':loop['passed_candidates'],
                'candidate_count':len(self.frozen_edges)+len(edges),
                'map_edges':self.frozen_edges + edges,
                'candidate_attempts':self.frozen_attempts+loop['candidate_attempts'],
                'descriptor_skips':self.frozen_skips+loop['descriptor_skips'],
                'descriptor_unindexed':self.frozen_unindexed+loop['descriptor_unindexed'],
                'submaps':self.submaps,'active_keyframes':len(part.frames),
                'frozen_keyframes':self.start,
                'timings_ms':{'descriptor':descriptor_ms,'registration_and_graph':graph_ms,
                    'map':(time.perf_counter()-tick)*1000,'total':(time.perf_counter()-began)*1000}}

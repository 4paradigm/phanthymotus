"""Native activity evidence, independent of graph growth or robot motion."""
import math

from .pointcloud_slam_graph import connected_nodes


class BackendProgress:
    def __init__(self, timeout_s=10.):
        if not isinstance(timeout_s,(int,float)) or isinstance(timeout_s,bool) or not math.isfinite(timeout_s) or timeout_s<=0:
            raise ValueError('positive finite backend timeout required')
        self.timeout_s=timeout_s
        self.pending_since=self.last_activity=None
        self.forwarded_stamp=self.activity_stamp=0
        self.graph_stamp=self.last_growth_stamp=0
        self.optimized_nodes=self.connected_nodes=self.last_growth=None
        self.new_connected_nodes=0
        self._seen_connected=set()

    def forward(self, stamp, now):
        self.forwarded_stamp=max(self.forwarded_stamp,stamp)
        if self.pending_since is None and self.forwarded_stamp>self.activity_stamp:
            self.pending_since=now

    def activity(self, stamp, now):
        if stamp<=self.activity_stamp:return
        self.activity_stamp,self.last_activity=stamp,now
        if stamp>=self.forwarded_stamp:self.pending_since=None

    def graph(self, stamp, now, nodes, links):
        if not nodes or stamp<=self.graph_stamp:return
        # The first graph is the baseline, including when resuming a saved map.
        anchors=self._seen_connected if self.graph_stamp else (min(nodes),)
        connected=connected_nodes(nodes,links,anchors)
        added=connected-self._seen_connected
        if self.graph_stamp and added:
            self.new_connected_nodes+=len(added)
            self.last_growth_stamp,self.last_growth=stamp,now
        self._seen_connected.update(added)
        self.graph_stamp=stamp
        self.optimized_nodes,self.connected_nodes=len(nodes),len(connected)
        self.activity(stamp,now)

    def status(self, now):
        state='idle'
        if self.pending_since is not None:
            if now-self.pending_since<=self.timeout_s:state='processing'
            elif self.last_activity is not None and now-self.last_activity<=self.timeout_s:state='processing_behind'
            else:state='suspected_stall'
        elif self.last_activity is not None:
            state='responsive' if self.last_growth_stamp==self.activity_stamp else 'responsive_without_new_nodes'
        # No per-frame consumed/skip acknowledgement exists in the pinned
        # transport. Silence alone is not enough evidence to kill its process.
        return dict(state=state,pending_since=self.pending_since,last_activity=self.last_activity,
                    forwarded_stamp_ns=self.forwarded_stamp,activity_stamp_ns=self.activity_stamp,
                    graph_stamp_ns=self.graph_stamp,optimized_nodes=self.optimized_nodes,
                    connected_nodes=self.connected_nodes,new_connected_nodes=self.new_connected_nodes,
                    last_growth_stamp_ns=self.last_growth_stamp,last_growth=self.last_growth,
                    automatic_restart_allowed=False,restart_budget=0)

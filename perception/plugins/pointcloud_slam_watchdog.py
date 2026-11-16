"""Native activity evidence, independent of graph growth or robot motion."""
import math


class BackendProgress:
    def __init__(self, timeout_s=10.):
        if not isinstance(timeout_s,(int,float)) or isinstance(timeout_s,bool) or not math.isfinite(timeout_s) or timeout_s<=0:
            raise ValueError('positive finite backend timeout required')
        self.timeout_s=timeout_s
        self.pending_since=self.last_activity=None
        self.forwarded_stamp=self.activity_stamp=0
        self.nodes=None
        self.grew=False

    def forward(self, stamp, now):
        self.forwarded_stamp=max(self.forwarded_stamp,stamp)
        if self.pending_since is None:self.pending_since=now

    def activity(self, stamp, now, *, nodes=None):
        if stamp<=self.activity_stamp:return
        self.activity_stamp,self.last_activity=stamp,now
        if stamp>=self.forwarded_stamp:self.pending_since=None
        self.grew=nodes is not None and (self.nodes is None or nodes>self.nodes)
        if nodes is not None:self.nodes=nodes

    def status(self, now):
        state='idle'
        if self.pending_since is not None:
            if now-self.pending_since<=self.timeout_s:state='processing'
            elif self.last_activity is not None and now-self.last_activity<=self.timeout_s:state='processing_behind'
            else:state='suspected_stall'
        elif self.last_activity is not None:
            state='responsive' if self.grew else 'responsive_without_new_nodes'
        # No per-frame consumed/skip acknowledgement exists in the pinned
        # transport. Silence alone is not enough evidence to kill its process.
        return dict(state=state,pending_since=self.pending_since,last_activity=self.last_activity,
                    forwarded_stamp_ns=self.forwarded_stamp,activity_stamp_ns=self.activity_stamp,
                    automatic_restart_allowed=False,restart_budget=0)

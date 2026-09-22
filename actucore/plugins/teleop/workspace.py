"""Conservative arm workspace checks shared by Tianyi and G1."""
import math
import numpy as np

class WorkspaceViolation(ValueError):
    """Preserve the public error code and identify its conservative sweep bound."""
    def __init__(self, code, sweep):
        super().__init__(code)
        self.sweep = sweep

def finite(value,shape):
    x=np.asarray(value,dtype=float)
    if x.shape!=shape or not np.isfinite(x).all():raise ValueError("invalid_finite_shape")
    return x

def segment_distance_lower(a, b, c, d):
    """Analytic finite-segment distance with a conservative roundoff allowance.

    Minimize the convex squared distance on [0,1]^2, clamping the edge
    optimum when the infinite-line solution lies outside the segments.
    The allowance covers almost-parallel cancellation; it can only reject
    extra geometry, never remove the configured capsule/sweep margins.
    """
    u=b-a;v=d-c;r=a-c
    uu=float(u@u);vv=float(v@v);uv=float(u@v)
    ur=float(u@r);vr=float(v@r)
    clamp=lambda x:max(0.,min(1.,x))
    if uu<=1e-24:
        x=0.;y=0. if vv<=1e-24 else clamp(vr/vv)
    elif vv<=1e-24:
        y=0.;x=clamp(-ur/uu)
    else:
        denominator=uu*vv-uv*uv
        x=clamp((uv*vr-ur*vv)/denominator) if denominator>0. else 0.
        y=(uv*x+vr)/vv
        if y<0.:y=0.;x=clamp(-ur/uu)
        elif y>1.:y=1.;x=clamp((uv-ur)/uu)
    offset=r+x*u-y*v
    return max(0.,math.sqrt(max(0.,float(offset@offset)))-1e-7*(1.+math.sqrt(uu)+math.sqrt(vv)))

class ArmWorkspace:
    def configure_workspace(self):
        self.workspace=self.profile['workspace']
        self._sampling_steps={}
        if not self.workspace.get('capsules') or not (self.workspace.get('torso_box') or self.workspace.get('torso_spheres')):
            raise ValueError('collision_calibration_missing')
        self.torso_spheres=[]
        for sphere in self.workspace.get('torso_spheres', []):
            center=finite(sphere['center_m'],(3,))
            radius=float(sphere['radius_m'])
            if not np.isfinite(radius) or not 0 < radius <= 1:
                raise ValueError('torso_sphere_calibration')
            self.torso_spheres.append((center,radius))
        self.capsules=[]
        for item in self.workspace['capsules']:
            a,b=(self.model.getFrameId(item[k]) for k in ('from','to'))
            radius=float(item['radius_m'])
            if max(a,b)>=self.model.nframes or not 0.01<=radius<=0.3:raise ValueError('capsule_calibration')
            self.capsules.append((a,b,radius,item['group']))
        def sweep_bounds(frame):
            if getattr(self,'axis_aware_sweep',False):
                coefficients=np.zeros(len(self.indices))
                joint=self.model.frames[frame].parentJoint
                center=np.array(self.model.frames[frame].placement.translation,copy=True)
                radius=0.
                axes={'JointModelRX':np.array([1.,0.,0.]),
                      'JointModelRY':np.array([0.,1.,0.]),
                      'JointModelRZ':np.array([0.,0.,1.])}
                while joint>0:
                    axis=axes.get(self.model.joints[joint].shortname())
                    if axis is None:break  # Unknown joint: retain the old global length bound.
                    wire=int(np.flatnonzero(self.indices==self.model.joints[joint].idx_q)[0])
                    radial=float(np.linalg.norm(np.cross(axis,center)))
                    coefficients[wire]=radial+radius+1e-12
                    # Rotate the enclosing ball through ALL joint angles. Its
                    # axial center stays fixed; the radial circle is enclosed by
                    # increasing the radius. Transform that ball to the parent.
                    parallel=axis*float(axis@center)
                    radius+=radial+1e-12
                    placement=self.model.jointPlacements[joint]
                    center=placement.translation+placement.rotation@parallel
                    joint=self.model.parents[joint]
                else:return coefficients
            coefficients=np.zeros(len(self.indices))
            joint=self.model.frames[frame].parentJoint
            distal=float(np.linalg.norm(self.model.frames[frame].placement.translation))
            while joint>0:
                wire=int(np.flatnonzero(self.indices==self.model.joints[joint].idx_q)[0])
                coefficients[wire]=distal
                distal+=float(np.linalg.norm(self.model.jointPlacements[joint].translation))
                joint=self.model.parents[joint]
            return coefficients
        self.sweep_coefficients=[np.maximum(sweep_bounds(a),sweep_bounds(b)) for a,b,_,_ in self.capsules]
        self.palm_sweep=[sweep_bounds(f) for f in self.frames]
        self._endpoint_paths={}
        for frame in set(self.frames)|{f for a,b,_,_ in self.capsules for f in (a,b)}:
            joint=self.model.frames[frame].parentJoint;path=[]
            while joint>0:
                axis={'JointModelRX':0,'JointModelRY':1,'JointModelRZ':2}.get(self.model.joints[joint].shortname())
                if axis is None:path=None;break
                wire=int(np.flatnonzero(self.indices==self.model.joints[joint].idx_q)[0])
                path.append((joint,wire,axis));joint=self.model.parents[joint]
            self._endpoint_paths[frame]=(path,sweep_bounds(frame))

    def _endpoint_excursion(self,frame,excursion):
        """Enclose the whole joint interval around the CURRENT FK pose.

        A ball (c,rho) rotated by nominal R plus +/-theta is contained in
        the ball (R*c, rho+2*sin(theta/2)*radial(c)). Existing uncertainty
        rotates without growing; rigid parent transforms preserve its radius.
        Induction from the endpoint to the root covers independent joints.
        """
        path,fallback=self._endpoint_paths[frame]
        if path is None:return float(fallback@excursion)
        center=self.model.frames[frame].placement.translation.copy();radius=0.
        for joint,wire,axis in path:
            radial=math.sqrt(max(0.,float(center@center)-float(center[axis])**2))
            radius+=2.*math.sin(min(math.pi,float(excursion[wire]))*.5)*radial+1e-12
            center=self.data.liMi[joint].act(center)
        return radius

    def _safe_transition(self, start, end, check_budget=lambda: None):
        """Cover the entire independently interpolated joint box, not its diagonal.

        Centered bounds halve the initial inflation. Ambiguous boxes are split;
        every child must pass. Exhausted refinement still rejects the movement.
        """
        start=finite(start,(len(self.indices),));end=finite(end,(len(self.indices),))
        pending=[(np.minimum(start,end),np.maximum(start,end),0,set())]
        while pending:
            check_budget()
            lo,hi,depth,proven=pending.pop();center=(lo+hi)*.5;radius=(hi-lo)*.5
            try:
                if getattr(self,"axis_aware_sweep",False):
                    self._safe_configuration(center,excursion=radius,proven=proven)
                else:self._safe_configuration(center,excursion=radius)
            except ValueError as exc:
                if str(exc) not in ('torso_collision','arm_collision','workspace_limit') or depth>=getattr(self,'transition_refinement_depth',4):
                    raise
                # Refine a joint that influences the failing geometry. A large
                # wrist motion cannot tighten an upper-arm/torso bound.
                scores=radius*getattr(exc,'sweep',np.ones_like(radius))
                axis=int(np.argmax(scores))
                if scores[axis]<=0:raise
                if radius[axis]<1e-7:raise
                left=hi.copy();left[axis]=center[axis]
                right=lo.copy();right[axis]=center[axis]
                pending.extend(((lo,left,depth+1,proven.copy()),(right,hi,depth+1,proven.copy())))
                # Consumed refinement signals must not retain configuration
                # frames through the violations list until cyclic GC runs.
                exc.__traceback__=None

    def _safe_configuration(self,q,excursion=None,proven=None):
        # A passed predicate covers the whole parent joint box and therefore
        # every child. Proofs live only inside this one transition traversal.
        if proven is None:proven=set()
        violations=[]
        model_q=np.empty(len(self.indices));model_q[self.indices]=q
        self.pin.framesForwardKinematics(self.model,self.data,model_q)
        torso=self.data.oMf[self.torso].inverse()
        local_bounds={};interval=excursion is not None and getattr(self,'axis_aware_sweep',False)
        def bound_for(frame):
            if frame not in local_bounds:local_bounds[frame]=self._endpoint_excursion(frame,excursion)
            return local_bounds[frame]
        # Capsule sampling is conservative: expand by half the point spacing.
        clouds={}
        analytic=getattr(self,"analytic_capsule_distance",False)
        box=None
        if self.workspace.get('torso_box'):
            box=finite(self.workspace['torso_box'],(2,3))
            if np.any(box[0]>=box[1]):raise ValueError('invalid_torso_box')
        needed=set()
        for i,(a,b,_,_) in enumerate(self.capsules):
            if (box is not None and ('box',i) not in proven) or any(('sphere',i,k) not in proven for k in range(len(self.torso_spheres))):needed.add(i)
            for j,(oa,ob,_,_) in enumerate(self.capsules[:i]):
                if not ({a,b}&{oa,ob}) and ('pair',i,j) not in proven:needed.update((i,j))
        for index,(a,b,r,group) in enumerate(self.capsules):
            if index not in needed:continue
            p0=(torso*self.data.oMf[a]).translation;p1=(torso*self.data.oMf[b]).translation
            delta=p1-p0
            length=math.sqrt(float(delta@delta));n=max(2,int(math.ceil(length/0.01))+1)
            # Per-ancestor lever-arm bounds cover every independent joint interpolation.
            extra_margin=0. if excursion is None else (max(bound_for(a),bound_for(b)) if interval else float(self.sweep_coefficients[index]@excursion))
            points=None
            if not analytic or box is not None:
                steps=self._sampling_steps.get(n)
                if steps is None:
                    steps=(np.arange(n,dtype=float)/(n-1))[:,None]
                    self._sampling_steps[n]=steps
                points=p0+steps*delta
            inflated=r+length/(2*(n-1))+0.005+extra_margin
            pair_radius=r+0.005+extra_margin if analytic else inflated
            if box is not None and ('box',index) not in proven:
                distance=np.linalg.norm(np.maximum(np.maximum(box[0]-points,points-box[1]),0),axis=1)
                if np.any(distance<inflated):violations.append(WorkspaceViolation('torso_collision',self.sweep_coefficients[index]))
                else:proven.add(('box',index))
            for sphere_index,(center,radius) in enumerate(self.torso_spheres):
                key=('sphere',index,sphere_index)
                if key in proven:continue
                # Exact segment/sphere distance needs no sampling inflation.
                t=0. if length==0 else min(1.,max(0.,float((center-p0)@delta)/(length*length)))
                offset=p0+t*delta-center
                if float(offset@offset)<(r+0.005+extra_margin+radius)**2+1e-14:
                    violations.append(WorkspaceViolation('torso_collision',self.sweep_coefficients[index]))
                else:proven.add(key)
            for other_index,(other,orr,oa,ob,other_bound) in clouds.items():
                key=('pair',index,other_index)
                if key in proven:continue
                # Only segments sharing an endpoint are adjacent; same-arm collisions count.
                if {a,b}&{oa,ob}:proven.add(key);continue
                collides=(segment_distance_lower(p0,p1,other[0],other[1])<pair_radius+orr+1e-12 if analytic else
                          np.min(np.sum(points*points,axis=1)[:,None]+np.sum(other*other,axis=1)[None,:]-2.*points@other.T)<(inflated+orr)**2+1e-14)
                if collides:violations.append(WorkspaceViolation('arm_collision',self.sweep_coefficients[index]+other_bound))
                else:proven.add(key)
            clouds[index]=((p0,p1) if analytic else points,pair_radius,a,b,self.sweep_coefficients[index])
        for side,f,bound in zip(('left','right'),self.frames,self.palm_sweep):
            key=('palm',side)
            if key in proven:continue
            limits=finite(self.workspace[side],(2,3));p=(torso*self.data.oMf[f]).translation
            margin=0. if excursion is None else (bound_for(f) if interval else float(bound@excursion))
            if np.any(p-margin<limits[0]) or np.any(p+margin>limits[1]):violations.append(WorkspaceViolation('workspace_limit',bound))
            else:proven.add(key)
        if violations:raise violations[0]

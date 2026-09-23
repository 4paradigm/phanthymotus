"""Fixed G1_23 upper-body geometry, including bounded joint sweeps."""
import hashlib
from itertools import combinations
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import hppfcl as fcl
import numpy as np


class G1Collision:
    def __init__(self, pin, model, urdf_path):
        self.pin, self.model = pin, model
        self.last_rejection = None
        root = ET.parse(urdf_path).getroot()
        assets = Path(__file__).parent/'models/g1_collision'
        hashes = json.loads((assets/'sha256.json').read_text())
        selected = {'torso_link'} | {s+'_'+part for s in ('left','right') for part in
            ('shoulder_pitch_link','shoulder_roll_link','shoulder_yaw_link','elbow_link','wrist_roll_rubber_hand')}
        self.objects = []
        for link in root.findall('link'):
            name = link.get('name')
            if name not in selected:continue
            collisions = link.findall('collision')
            if len(collisions)!=1:raise ValueError('g1_collision_geometry_missing')
            collision = collisions[0];origin = collision.find('origin')
            placement = pin.SE3(pin.rpy.rpyToMatrix(np.fromstring(origin.get('rpy'),sep=' ')),
                                np.fromstring(origin.get('xyz'),sep=' '))
            mesh = collision.find('geometry/mesh')
            if mesh is not None:
                filename = Path(mesh.get('filename')).name
                path = assets/filename
                if hashlib.sha256(path.read_bytes()).hexdigest()!=hashes[filename]:
                    raise ValueError('g1_collision_mesh_changed')
                loaded = fcl.MeshLoader().load(str(path))
                loaded.buildConvexHull(True,'Qt')
                shape = loaded.convex
            else:
                cylinder = collision.find('geometry/cylinder')
                if cylinder is None:raise ValueError('g1_collision_geometry_unsupported')
                shape = fcl.Cylinder(float(cylinder.get('radius')),float(cylinder.get('length')))
            frame_id = model.getFrameId(name)
            if frame_id>=model.nframes:raise ValueError('g1_collision_frame_missing')
            shape.computeLocalAABB()
            # Every point lies within this radius about the geometry origin.
            radius = float(np.linalg.norm(shape.aabb_center)+shape.aabb_radius)
            frame = model.frames[frame_id]
            distal = radius+float(np.linalg.norm((frame.placement*placement).translation))
            joint = frame.parentJoint;bound = np.zeros(model.nq)
            while joint>0:
                bound[model.joints[joint].idx_q] = distal
                distal += float(np.linalg.norm(model.jointPlacements[joint].translation))
                joint = model.parents[joint]
            self.objects.append((name,frame_id,placement,shape,bound))
        if {o[0] for o in self.objects}!=selected:raise ValueError('g1_collision_links_missing')
        adjacent = {frozenset((j.find('parent').get('link'),j.find('child').get('link')))
                    for j in root.findall('joint')}
        # Only mechanically adjacent links are excluded; all other 45 pairs remain.
        self.pairs = [(a,b) for a,b in combinations(range(len(self.objects)),2)
                      if frozenset((self.objects[a][0],self.objects[b][0])) not in adjacent]

    def check(self, data, excursion=None):
        poses = []
        for _,frame,placement,_,_ in self.objects:
            pose = data.oMf[frame]*placement
            poses.append(fcl.Transform3f(pose.rotation,pose.translation))
        # A world-space bounding sphere encloses each complete geometry. Only
        # skip a narrow-phase query when the spheres prove the full swept
        # clearance; near pairs still use the original exact distance check.
        centers = np.array([data.oMf[frame].act(placement.act(shape.aabb_center))
                            for _,frame,placement,shape,_ in self.objects])
        radii = np.array([shape.aabb_radius for _,_,_,shape,_ in self.objects])
        pairs = np.asarray(self.pairs)
        lower = np.linalg.norm(centers[pairs[:,0]]-centers[pairs[:,1]],axis=1)
        lower -= radii[pairs[:,0]]+radii[pairs[:,1]]
        bounds = np.array([obj[4] for obj in self.objects])
        sweeps = np.zeros(len(pairs)) if excursion is None else (bounds[pairs[:,0]]+bounds[pairs[:,1]])@excursion
        for index,(a,b) in enumerate(self.pairs):
            left,right = self.objects[a],self.objects[b]
            # Sum of ancestor arc-length bounds encloses the entire joint interpolation.
            swept = float(sweeps[index])
            if np.isfinite(lower[index]) and lower[index] > .005+swept+1e-9:
                continue
            distance = fcl.distance(left[3],poses[a],right[3],poses[b],
                                    fcl.DistanceRequest(),fcl.DistanceResult())
            if not np.isfinite(distance) or distance<=.005+swept:
                self.last_rejection = {'pair':[left[0],right[0]],'distance_m':float(distance) if np.isfinite(distance) else None,'swept_margin_m':swept,'clearance_m':.005,'kind':'static' if excursion is None else 'swept_bound'}
                raise ValueError('g1_body_collision:'+left[0]+':'+right[0])

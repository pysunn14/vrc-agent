"""Model-free initial pose and explicit room-axis adjustments for calibration."""
from copy import deepcopy
import math
from .asset_contract import pose_data, vector, number


def multiply(a, b):
    x,y,z,w=a; X,Y,Z,W=b
    q=(w*X+x*W+y*Z-z*Y,w*Y-x*Z+y*W+z*X,w*Z+x*Y-y*X+z*W,w*W-x*X-y*Y-z*Z)
    norm=math.sqrt(sum(v*v for v in q))
    return [v/norm for v in q]


def candidate(rig, hmd_base):
    base=vector(hmd_base,3)
    if base[1]<=0: raise ValueError('head origin height must be positive')
    scale=rig.scale_for_hmd_height(base[1])
    # The existing neutral generator uses this slightly bent, downward ray.
    # This is an editable starting point, never a claim of controller-mount calibration.
    ray=(.25,.94,.20); norm=math.sqrt(sum(v*v for v in ray))
    reach=rig.arm_reach_m*scale*.85
    trackers={}
    for side,sign in (('left',-1),('right',1)):
        shoulder=rig.bone(side+'UpperArm').position_m
        pos=[base[i]+shoulder[i]*scale for i in range(3)]
        delta=(sign*ray[0],-ray[1],ray[2])
        trackers[side]={'position':[pos[i]+delta[i]*reach/norm for i in range(3)],
                        'quaternion_xyzw':[math.sqrt(.5),0,0,math.sqrt(.5)]}
    for key,bone in (('hips','hips'),('left_foot','leftFoot'),('right_foot','rightFoot')):
        p=rig.body_tracker_poses(hmd_base=base)[bone]
        trackers[key]={'position':list(p.position_m),'quaternion_xyzw':list(p.rotation_xyzw)}
    return pose_data(dict(kind='six_point_pose',head={'xyz_cm':[0,0,0],'ypr_deg':[0,0,0]},
        trackers=trackers,fps=20,scale=scale,locomotion=[0,0,0],
        tracker_activation={key:True for key in trackers},face={}))


def adjust(pose, hand, axis, delta):
    pose_data(pose)
    if hand not in ('left','right','both'): raise ValueError('hand must be left, right or both')
    if axis not in ('x','y','z','rx','ry','rz','spread'): raise ValueError('unknown adjustment axis')
    number(delta,-30 if axis.startswith('r') else -.10,30 if axis.startswith('r') else .10)
    result=deepcopy(pose)
    for side in ('left','right') if hand=='both' else (hand,):
        tracker=result['trackers'][side]
        if axis.startswith('r'):
            angle=math.radians(delta)/2
            q=[0,0,0,math.cos(angle)]; q['xyz'.index(axis[1])]=math.sin(angle)
            tracker['quaternion_xyzw']=multiply(q,tracker['quaternion_xyzw'])
        else:
            index=0 if axis=='spread' else 'xyz'.index(axis)
            tracker['position'][index]+=delta*(-1 if side=='left' else 1) if axis=='spread' else delta
    return pose_data(result)

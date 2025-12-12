#Python reference: Quaternions: https://stackoverflow.com/questions/39000758/how-to-multiply-two-quaternions-by-python-or-numpy
# 
##comment V.2:min comment check GDRIVE DEc/7 for full explanation vers
#imports
from flask import Flask, render_template, request, jsonify
import numpy as np


app = Flask(__name__)


def q_from_axis_angle(axis, angle):
   axis =axis/ (np.linalg.norm(axis) + 1e-12)
   s= np.sin(angle/2.0)
   return np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(angle/2.0)], dtype=float)

def q_mul(a, b):
   ax,ay,az,aw = a
   bx,by,bz,bw = b
   x = aw*bx + ax*bw + ay*bz - az*by
   y = aw*by - ax*bz + ay*bw + az*bx
   z = aw*bz + ax*by - ay*bx + az*bw
   w = aw*bw - ax*bx - ay*by - az*bz
   return np.array([x,y,z,w], dtype=float)


def q_conjugate(q):
   return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)
def q_normalize(q):
   return q/(np.linalg.norm(q) + 1e-12)
def q_rotate(q, v):
   qv = np.array([v[0], v[1], v[2], 0.0], dtype=float)
   return q_mul(q_mul(q, qv), q_conjugate(q))[:3]


plane_state = {
   'position':np.array([0.0, 60.0, 0.0], dtype=float),
   'velocity':np.array([140.0, 0.0, 0.0], dtype=float),   ###world frame
   'quaternion':np.array([0.0,0.0,0.0,1.0], dtype=float),  
   'throttle':0.6
}



g = 9.8
dt = 1.0/30.0


       #Scale for inptuts-->adjust
rot_deg_per_input=0.35    
stability_restore=0.45  
max_pitch_deg=35.0   #pitch not more thna 35


#terrain
def terrain_height(x,z):
   return 5.0 +8.0*np.sin(0.0007*x)*np.cos(0.0008*z)


min_alt_above_ground=6.0


#reset endpoint
@app.route('/reset', methods=['POST'])
def reset():
   plane_state['position']=np.array([0.0, 60.0, 0.0], dtype=float)
   plane_state['velocity']=np.array([140.0, 0.0, 0.0], dtype=float)
   plane_state['quaternion']=np.array([0.0,0.0,0.0,1.0], dtype=float)
   plane_state['throttle']=0.6
   return jsonify({'status':'ok'})


@app.route('/')
def index():
   return render_template('index.html')


@app.route('/update', methods=['POST'])
def update():
   data=request.json or {}
   
   pitch_in=float(data.get('pitch', 0.0))   
   yaw_in=float(data.get('yaw', 0.0))
   roll_in=float(data.get('roll', 0.0))
   thr_in=float(data.get('throttle', plane_state['throttle']))


   # clamp & apply throttle
   thr_in=float(np.clip(thr_in, 0.0, 1.0))
   plane_state['throttle']=thr_in


  
   dp=np.radians(np.clip(pitch_in, -1.0, 1.0) * rot_deg_per_input) 
   dy=np.radians(np.clip(yaw_in, -1.0, 1.0) * rot_deg_per_input)
   dr =np.radians(np.clip(roll_in, -1.0, 1.0) * rot_deg_per_input)



   q= q_normalize(plane_state['quaternion'])
   forward =q_rotate(q, np.array([1.0,0.0,0.0]))
   up=q_rotate(q, np.array([0.0,1.0,0.0]))
   right= q_rotate(q, np.array([0.0,0.0,1.0]))


  
   world_up=np.array([0.0,1.0,0.0])
   pitch_angle=np.degrees(np.arcsin(np.clip(np.dot(forward, world_up), -0.999, 0.999)))


  
   if pitch_angle > max_pitch_deg and pitch_in < 0:  
       dp =0.0
   if pitch_angle < -max_pitch_deg and pitch_in > 0:
       dp=0.0


   R_yaw =q_from_axis_angle(up, dy)
   R_pitch =q_from_axis_angle(right, dp)
   R_roll =q_from_axis_angle(forward, dr)

   q_new=q_mul(q, q_mul(R_yaw, q_mul(R_pitch, R_roll)))
   q_new=q_normalize(q_new)
  
   roll_angle=np.degrees(np.arctan2(np.dot(right, world_up), np.dot(up, world_up)))


   restore_pitch= -np.radians(pitch_angle * stability_restore * 0.02)  #tiny
   restore_roll= -np.radians(roll_angle  * stability_restore * 0.02)

   ##FIXING ThE ROTATIONS
   R_restore_pitch= q_from_axis_angle(right, restore_pitch)
   R_restore_roll  =q_from_axis_angle(forward, restore_roll)
   q_final=q_mul(q_new, q_mul(R_restore_pitch, R_restore_roll))
   q_final=q_normalize(q_final)


   plane_state['quaternion'] =q_final
   base_speed=140.0
   speed=base_speed * (0.6 + 0.8 * plane_state['throttle'])#0.6..1.4*base
   forward_world=q_rotate(q_final, np.array([1.0,0.0,0.0]))
   forward_world /= (np.linalg.norm(forward_world) + 1e-12)


   # small 
   lift_up=6.0 * (np.cos(np.radians(pitch_angle)))  ##tuned constant


   # compute velocity vector: forward * speed + small upward lift term
   vel_target=forward_world * speed + np.array([0.0, lift_up, 0.0])
   plane_state['velocity'] += (vel_target - plane_state['velocity']) * 0.12
   #"integration"
   plane_state['position'] += plane_state['velocity'] * dt


   #terrain backend?
   tx, tz=plane_state['position'][0], plane_state['position'][2]
   h =terrain_height(tx, tz) + min_alt_above_ground
   if plane_state['position'][1] < h:
       plane_state['position'][1]= h
       if plane_state['velocity'][1] < 0:
           plane_state['velocity'][1]=0.0


   
   acc=(vel_target - plane_state['velocity']) / (dt + 1e-12)
   # include gravity
   acc_world=acc + np.array([0.0, -g, 0.0])


   # Gspine
   up_world=q_rotate(q_final, np.array([0.0,1.0,0.0]))
   up_world /= (np.linalg.norm(up_world) + 1e-12)
   g_spine=float(np.dot(acc_world, up_world)/g)


   # limit g dispkay
   g_spine = float(np.clip(g_spine, -6.0, 9.0))


   qf = q_normalize(plane_state['quaternion'])
   return jsonify({
       'x':float(plane_state['position'][0]),
       'y':float(plane_state['position'][1]),
       'z':float(plane_state['position'][2]),
       'qx': float(qf[0]),
       'qy': float(qf[1]),
       'qz': float(qf[2]),
       'qw': float(qf[3]),
       'g': round(g_spine,2),
       'speed': float(np.linalg.norm(plane_state['velocity'])),
       'throttle': round(plane_state['throttle'], 2)
   })


if __name__ == '__main__':
   app.run(debug=True)




#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse
import cv2.load_config_py2
import pyglet
from pyglet.gl import *
import numpy as np
import cv2
from pyglet.window import key
import gym
from gym_duckietown.envs import DuckietownEnv
import Tarea3_utils_PathPlanning as tpp
import math
import io
import requests
import json

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='Duckietown_udem1')
parser.add_argument('--map-name', default='proyecto_final')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()

if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed = args.seed,
        map_name = args.map_name,
        draw_curve = args.draw_curve,
        draw_bbox = args.draw_bbox,
        domain_rand = args.domain_rand,
        frame_skip = args.frame_skip,
        distortion = args.distortion,
    )
else:
    env = gym.make(args.env_name)
env.reset()
env.render()
env.unwrapped.cam_angle[0] = 0


#-----------------------------------------
data = []
for i in env.grid:
    data.append([i["coords"], i["kind"], i["drivable"]])

matriz,data = tpp.data_map(data)
matrix_color = np.zeros((matriz.shape[0], matriz.shape[1], 3), dtype=np.uint8)
# Asignar colores: negro para 0 y blanco para 1
matrix_color[matriz == 0] = [0, 0, 0]
matrix_color[matriz == 1] = [255, 255, 255]
# el valor 1 es para las posiciones donde se puede conducir
PointB = tpp.randomPoint(matriz)
cv2.namedWindow("Path_Planning", cv2.WINDOW_NORMAL)


def get_pos():
    n = 2
    road_tile_size = env.road_tile_size
    x,y = env.cur_pos[0],env.cur_pos[2]
    x0 = x//road_tile_size # posicion en x en la matriz original
    y0 = y//road_tile_size # posicion en y en la matriz original
    x1 = x-x0*road_tile_size # posicion en x en la matriz de n
    y1 = y-y0*road_tile_size # posicion en y en la matriz de n
    if x1 >= road_tile_size/2:
        x1 = 1
    else:
        x1 = 0
    if y1 >= road_tile_size/2:
        y1 = 1
    else:
        y1 = 0
    return (int(y0*n+y1),int(x0*n+x1))


class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0, tolerance=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.tolerance = tolerance
        self.previous_error = 0
        self.integral = 0

    def compute(self, measurement):
        # Ajuste del error basado en el rango del setpoint
        if self.setpoint - self.tolerance <= measurement <= self.setpoint + self.tolerance:
            error = 0
        else:
            error = self.setpoint - measurement
        
        self.integral += error
        derivative = error - self.previous_error
        self.previous_error = error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return output
    
    def reset(self):
        self.previous_error = 0
        self.integral = 0
    
    def set_setpoint(self, setpoint):
        self.setpoint = setpoint


PointA = get_pos()
path,_ = tpp.Path(data,PointA,PointB)

def update_Path(dt):
    global matrix
    global data
    global path
    global PointA
    global PointB
    PointA = get_pos()
    path,_ = tpp.Path(data,PointA,PointB)
    

def Path_planning():
    global matriz
    global matrix_color
    global data
    global PointA,PointB
    
    copia = matrix_color.copy()    
    
    if path is not None:
        for i in path:
            x,y = data.loc[data['node'] == i, 'pos'].values[0]
            copia[x,y] = (150, 0, 0)
    
    copia[PointB] = (250, 0, 255)
    copia[PointA] = (250, 0, 0)
    cv2.imshow("Path_Planning", copia)


flag_init = 0
data_init = [] # pos inicial
Alert = False
def Motion_Planning():
    global data
    global PointA,PointB
    global path
    global data_init
    global Alert
    global matriz
    """
        obtengo el path con los nodos que debo seguir para llegar al destino, guardo los 3 primeros nodos y los elimino de la lista
    """
    # ajustar posicion inicial
    # si avanzo y no hay camino, mi direccion es contraria a la que deberia ser, busco el camino mas cercano y me muevo hacia ahi
    if path is not None:
        # elimino mi posicion actual

        if Alert:
            for i in range(3):
                if len(path) < i+1:
                    break

                type_sector = data.loc[data['node'] == path[i], 'type_sector'].values[0]
                pos = data.loc[data['node'] == path[i], 'pos'].values[0]
                if type_sector == 1 or type_sector != 2:
                    if matriz[pos[0],pos[1]+1] == 1:
                        new_pos = (pos[0],pos[1]+1)
                        path[i] = data.loc[data['pos'] == new_pos, 'node'].values[0]
                    else:
                        new_pos = (pos[0],pos[1]-1)
                        path[i] = data.loc[data['pos'] == new_pos, 'node'].values[0]
                if type_sector == 2 or type_sector != 1:
                    if matriz[pos[0]+1,pos[1]] == 1:
                        new_pos = (pos[0]+1,pos[1])
                        path[i] = data.loc[data['pos'] == new_pos, 'node'].values[0]
                    else:
                        new_pos = (pos[0]-1,pos[1])
                        path[i] = data.loc[data['pos'] == new_pos, 'node'].values[0]
            Alert = False
            
        try:
            node = data.loc[data['pos'] == PointA, 'node'].values[0]
            path.remove(node)
        except:
            aux = 0

        if len(path) > 0:
            if len(path) >= 3:
                data_init = path[0:3]
            else:
                data_init = path 

        else:
            data_init = []
            path = None



# definimos nuestras posibles direcciones

def transfor_poss(x,y):
    road_tile_size = env.road_tile_size
    return (((2*x+1)*road_tile_size)/4,(((2*y)+1)*road_tile_size)/4)

def angle(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    # Calcular el ángulo en radianes
    angle = math.atan2(dy, dx)
    angle *= -1
    return angle

last_error = 0
def control_pid_angulo(angulo_actual, angulo_deseado):
    global last_error
    Kp = 4.0
    Ki = 0.01
    Kd = 0.05
    error = angulo_deseado - angulo_actual
    if error > math.pi:
        error -= 2 * math.pi
    elif error < -math.pi:
        error += 2 * math.pi
    P = Kp * error
    I = Ki * error
    D = Kd * (error - last_error)
    last_error = error
    control = P + I + D
    return control


def automatic_move():
    # mantiene al vehiculo en el centro del camino.
    global data
    global data_init
    global PointA
    global Alert
    if data_init is [] or data_init is None or len(data_init) == 0:
        return np.array([0.0, 0.0])
    if len(data_init) == 0:
        return np.array([0.0, 0.0])
    else:
        new_pos = data.loc[data['node'] == data_init[0], 'pos'].values[0]
        all_pos = [data.loc[data['node'] == x, 'pos'].values[0] for x in data_init]
        if all(x[0] == all_pos[0][0] for x in all_pos) and PointA[0] == all_pos[0][0]:
            new_pos = data.loc[data['node'] == data_init[-1], 'pos'].values[0]   
        elif all(x[1] == all_pos[0][1] for x in all_pos) and PointA[1] == all_pos[0][1]:
            new_pos = data.loc[data['node'] == data_init[-1], 'pos'].values[0]   
        x2,y2 = transfor_poss(new_pos[1],new_pos[0])
        current_velocity = env.speed
        current_angle = env.cur_angle
        x,y = env.cur_pos[0],env.cur_pos[2]
        
        set_angle = angle(x, y, x2, y2)
        angle_pid = control_pid_angulo(current_angle, set_angle)
        angle_set = np.clip(angle_pid, -3, 3)
        
        speed_pid = PID(Kp=1.0, Ki=0.0, Kd=0.6, setpoint=2, tolerance=0.05)
        speed_correction = speed_pid.compute(current_velocity)
        
        if angle_set >= -0.2 and angle_set <= 0.2:
            speed_set = np.clip(speed_correction, 0, 1)
        else:
            speed_set = np.clip(speed_correction, 0, 0.2)
        action = np.array([speed_set, angle_set])
        #return np.array([0.0, 0.0])
        return action
    


def get_prediction(image):
    
    _, img_encoded = cv2.imencode('.jpg', image)
    image_bytes = io.BytesIO(img_encoded)

    url = "http://127.0.0.1:8000/detect/"
    files = {'file': image_bytes}
    r = requests.post(url, files=files)
    response =json.loads(r.text)
    response =json.loads(response["detections"])
    return response
#-----------------------------------------
#utils 
def lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    p = np.array([x1, y1])
    r = np.array([x2 - x1, y2 - y1])
    q = np.array([x3, y3])
    s = np.array([x4 - x3, y4 - y3])
    r_cross_s = np.cross(r, s)
    q_minus_p = q - p
    q_minus_p_cross_r = np.cross(q_minus_p, r)
    if r_cross_s == 0 and q_minus_p_cross_r == 0:
        t0 = np.dot(q_minus_p, r) / np.dot(r, r)
        t1 = t0 + np.dot(s, r) / np.dot(r, r)
        return (0 <= t0 <= 1) or (0 <= t1 <= 1)
    if r_cross_s == 0 and q_minus_p_cross_r != 0:
        return False
    t = np.cross(q_minus_p, s) / r_cross_s
    u = np.cross(q_minus_p, r) / r_cross_s
    return (0 <= t <= 1) and (0 <= u <= 1)

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """
    global PointA,PointB
    global path
    global data_init
    if symbol == key.BACKSPACE or symbol == key.SLASH or path is None or len(path) == 0:
        env.reset()
        env.render()
        print('RESET')
        
        PointB = tpp.randomPoint(matriz)
        PointA = get_pos()
        update_Path(0)
        print("nuevo destino: ",PointB)
        data_init = []
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)
intersect = False
def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    global PointA
    global PointB
    global path
    global data_init
    global Alert 
    global intersect
    PointA = get_pos()
    action = automatic_move()  
    if key_handler[key.UP]:
        action = np.array(object=[0.44, 0.0])
    if key_handler[key.DOWN]:
        action = np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        action = np.array([0.35, +1])
    if key_handler[key.RIGHT]:
        action = np.array([0.35, -1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])
    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5
    if path is None or len(path) == 0:
        PointB = tpp.randomPoint(matriz)
        PointA = get_pos()
        update_Path(0)
        print("nuevo destino: ",PointB)
        data_init = []
    obs, reward, done, info = env.step(action)
    duckies_detected = get_prediction(obs) # custom dataset
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
    
    #print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

    #colocar lineas en el centro inferior de la imagen
    
    cv2.line(obs, (int((3*obs.shape[1])/4),int(obs.shape[0]*0.78)), (int((obs.shape[1])/4),int(obs.shape[0]*0.78)), (0, 0, 255), 2)
    limit_x1 = int((3*obs.shape[1])/4)+70
    limit_y1 = int(obs.shape[0]*0.77)
    limit_x2 = int((obs.shape[1])/4)-30
    limit_y2 = int(obs.shape[0]*0.77)
    
    for detection in duckies_detected:
        confidence = detection["confidence"]
        confidence = round(confidence, 2)
        if confidence < 0.5:
            continue
        x1 = detection["box"]["x1"]
        y1 = detection["box"]["y1"]
        x2 = detection["box"]["x2"]
        y2 = detection["box"]["y2"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        line_x = int((x1 + x2)/2)
        if detection["name"] == "Duckies":
            cv2.line(obs, (line_x, y1), (line_x, y2), (0, 255, 0), 2)
            intersect = lines_intersect(line_x, y1, line_x, y2, limit_x1, limit_y1, limit_x2, limit_y2)
        if intersect:
            print('Duckie detected in via')
            Alert = True
        else:
            Alert = False
            intersect = False


        if detection["name"] == "Duckies":
            cv2.rectangle(obs, (x1, y1), (x2, y2), (255, 0, 0), 2)
        else:
            cv2.rectangle(obs, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        name = detection["name"]
        cv2.putText(obs, name, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, 1)
        cv2.putText(obs, str(confidence), (x1, y1+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, 1)

    cv2.imshow("obs", obs)
    if done:
        #print('done!')
        #env.reset()
        #env.render()
        pass
    
    # print(env.cur_pos) [3.83193618 0.         2.66608419]
    # print(env.cur_angle) 3.1352243456171034
    # print(env.grid_width) 17 
    # print(env.grid_height) 9
    # print(env.road_tile_size) 0.585

 
    
    cv2.waitKey(1)
    env.render()
    Path_planning()
    Motion_Planning()

pyglet.clock.schedule_interval(update, 1 / env.unwrapped.frame_rate)
#pyglet.clock.schedule_interval(update_Path, 20 / env.unwrapped.frame_rate)
# Enter main event loop
pyglet.app.run()

env.close()

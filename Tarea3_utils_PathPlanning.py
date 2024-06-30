import pandas as pd
import numpy as np
import heapq
import numpy as np
import random
import pandas as pd
import heapq
import cv2

def invert_tuple(t):
    return (t[1], t[0])

def get_index(data, point):
    for i in range(len(data)):
        if data[i] == point:
            return i
    return -1

def gen_map_data(n:int, data:list):
    """
        gen_map_data: Genera un diccionario con la informacion de los sectores.\n
        entry: \n
            int n -> escala de aumento de la matriz \n
            list data -> lista de tuplas con la informacion de los sectores \n
        output: dict
        
    """
    map_data = {}
    map_data["upscale"] = n
    data_driving = [i[0] for i in data if i[2] == 1]
    data_driving = [invert_tuple(i) for i in data_driving]
    data_driving.sort()
    map_data["sectors"] = data_driving
    map_data["num_sectors"] = len(data_driving)
    map_data["type"] = {1:"top_down", 2:"left_right", 3:"corner", 4:"bifurcation"}
    map_data["points"] = []
    for i in range(len(data_driving)):
        neighbors_directions = [(0,1), (1,0), (0,-1), (-1,0)]
        neighbors = []
        type_sector = list((0,0,0,0))
        for direction in neighbors_directions:
            new_position = (data_driving[i][0] + direction[0], data_driving[i][1] + direction[1])
            if new_position in data_driving:
                neighbors.append(new_position)
                type_sector[3] += 1
                if direction == (0,1) or direction == (0,-1):
                    type_sector[1] += 1
                if direction == (1,0) or direction == (-1,0):
                    type_sector[0] += 1
                if direction == (1,0) or direction == (0,1):
                    type_sector[2] += 1
                if direction == (-1,0) or direction == (0,-1):
                    type_sector[2] += 1
        aux = max(enumerate(type_sector), key=lambda x: x[1])
        if aux[1] >= 3:
            type_sector = 4
        else:
            type_sector = aux[0]+1
        neighbors = [get_index(data_driving, i) for i in neighbors]
        map_data[i] = {"obstacle":False,"neighbors":neighbors, "type":type_sector,"points":[]}
    return map_data

def upscale(n,data):
    positions = [i[0] for i in data]
    x = max(positions, key=lambda x: x[0])[0]
    y = max(positions, key=lambda x: x[1])[1]
    matriz = np.zeros((y+1, x+1))
    for i in data:
        matriz[i[0][1], i[0][0]] = i[2]
    map_data = gen_map_data(2, data)
    new_matriz = np.zeros((matriz.shape[0]*n, matriz.shape[1]*n))
    sector = 0
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            new_matriz[i*n:(i+1)*n, j*n:(j+1)*n] = matriz[i,j]
            if matriz[i,j] == 1:
                for k in range(i*n,(i+1)*n):
                    for m in range(j*n,(j+1)*n):
                        map_data[sector]['points'].append((k,m))
                        map_data["points"].append((k,m))
                sector += 1
    return new_matriz, map_data

def connect_sectors(n, map_data,matriz):
  if n == 1:
    # retorna grafo sin direcciones y con sus respectivos pesos para cada nodo
    num = map_data["num_sectors"]
    node = [i for i in range(num)]
    pos = [list(map_data[i]["points"]) for i in range(num)]
    sectores = [i for i in range(num)]
    weight = [map_data[i]["type"] for i in range(num)]
    edges = [map_data[i]["neighbors"] for i in range(num)]
    type_sector = map_data["type"]
    grafo = pd.DataFrame({"sectores":sectores,"pos":pos,"node":node, "weight":weight, "edges":edges,"type_sector":type_sector}, index=node)
    return grafo
  if n == 2:
    columns = ["sectores","pos","node", "weight", "edges","type_sector"]
    row = []
    node=0
    for i in range(map_data["num_sectors"]):
      for j in map_data[i]["points"]:
        aux = [i,j,node]
        node += 1
        node_type = 1
        neighbors = []
        
        if map_data[i]["type"] == 1:# top_down
          node_type = 1
          if (j[0],j[1]+1) in map_data["points"]:
            neighbors.append((j[0]+1,j[1]))
          if (j[0],j[1]-1) in map_data["points"]:
            neighbors.append((j[0]-1,j[1]))
        
        if map_data[i]["type"] == 2:# left_right
          node_type = 1
          if (j[0]+1,j[1]) in map_data["points"]:
            neighbors.append((j[0],j[1]-1))
          if (j[0]-1,j[1]) in map_data["points"]:
            neighbors.append((j[0],j[1]+1))
        
        if map_data[i]["type"] == 3:# corner
          node_type = 2
          if (j[0]-1,j[1]) not in map_data[i]["points"]:
            if (j[0],j[1]-1) not in map_data[i]["points"]:
              if (j[0],j[1]-1) in map_data["points"]:
                neighbors.append((j[0],j[1]-1))
              else:
                neighbors.append((j[0]+1,j[1]))
            else:
              if (j[0]-1,j[1]) in map_data["points"]:
                neighbors.append((j[0]-1,j[1]))
              else:
                neighbors.append((j[0],j[1]-1))
          else:
            if (j[0],j[1]+1) not in map_data[i]["points"]:
              if (j[0],j[1]+1) in map_data["points"]:
                neighbors.append((j[0],j[1]+1))
              else:
                neighbors.append((j[0]-1,j[1]))
            else:
              if (j[0]+1,j[1]) in map_data["points"]:
                neighbors.append((j[0]+1,j[1]))
              else:
                neighbors.append((j[0],j[1]+1))
               
        if map_data[i]["type"] == 4:# bifurcation
          node_type = 4
          if (j[0]-1,j[1]) not in map_data[i]["points"]:
            if (j[0],j[1]-1) not in map_data[i]["points"]:
              neighbors.append((j[0]+1,j[1]))
              if (j[0],j[1]-1) in map_data["points"]:
                neighbors.append((j[0],j[1]-1))
            else:
              neighbors.append((j[0],j[1]-1))
              if (j[0]-1,j[1]) in map_data["points"]:
                neighbors.append((j[0]-1,j[1]))
          else:
            if (j[0],j[1]-1) not in map_data[i]["points"]:
              neighbors.append((j[0],j[1]+1))
              if (j[0]+1,j[1]) in map_data["points"]:
                neighbors.append((j[0]+1,j[1]))
            else:
              neighbors.append((j[0]-1,j[1]))
              if (j[0],j[1]+1) in map_data["points"]:
                neighbors.append((j[0],j[1]+1))
              

        aux.append(node_type)  
        aux.append(neighbors)
        aux.append(map_data[i]["type"])  
        row.append(aux)
    
    grafo = pd.DataFrame(row, columns=columns)
    
    return grafo

# Función heurística (distancia Manhattan)
def heuristic(data, a, b):
    a_pos = data[data["node"] == a]["pos"].values[0]
    b_pos = data[data["node"] == b]["pos"].values[0]
    return abs(a_pos[0] - b_pos[0]) + abs(a_pos[1] - b_pos[1])

# Algoritmo A* optimizado
def Astar(data, start, end):
    # Cola de prioridad
    frontier = []
    heapq.heappush(frontier, (0, start, 0, 0, []))  # (priority, position, consecutive_count, current_cost, path)
    
    cost_so_far = {start: 0}
    visited = set()
    
    while frontier:
        _, actual_pos, consecutive_count, current_cost, path = heapq.heappop(frontier)
        
        if actual_pos == end:
            return path + [end], current_cost
        
        if (actual_pos, consecutive_count) in visited:
            continue
        
        visited.add((actual_pos, consecutive_count))
        
        neighbors = data[data["node"] == actual_pos]["edges"].values[0]
        neighbors = [data[data["pos"] == i]["node"].values[0] for i in neighbors]
        
        for next_node in neighbors:
            weight = data[data["node"] == next_node]["weight"].values[0]
            new_cost = current_cost + weight
            new_consecutive_count = consecutive_count + 1 if weight == 4 else 0
            
            if weight == 4 and new_consecutive_count > 3:
                continue
            
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(data, end, next_node)
                heapq.heappush(frontier, (priority, next_node, new_consecutive_count, new_cost, path + [actual_pos]))
    
    return None, float('inf')

def randomPoint(matriz):
  points = np.column_stack(np.where(matriz == 1))
  dest = random.choice(points)
  return (dest[0],dest[1])

def Path(data,pointA,pointB):
  try:
    start_node = data.loc[data["pos"] == (pointA[0],pointA[1])]["node"].values[0]
    end_node = data.loc[data["pos"] == (pointB[0],pointB[1])]["node"].values[0]
    path,cost = Astar(data,start_node, end_node)
    if cost == float('inf'):
      return None,None
    return path,cost
  except Exception as e:
    print(e)
    return None,None
    

def data_map(data):
    positions = [i[0] for i in data]
    x = max(positions, key=lambda x: x[0])[0]
    y = max(positions, key=lambda x: x[1])[1]
    del positions
    matriz = np.zeros((y+1, x+1))
    for i in data:
        matriz[i[0][1], i[0][0]] = i[2]
    n = 2
    matriz,map_data = upscale(n, data)
    data = connect_sectors(n, map_data,matriz)
    return matriz,data
  

# buscar el duckiebot en la imagen aerea
def get_Duckiebot(img):
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    range_low = np.array([100, 180, 120])  # Rango bajo de tono (rojo)
    range_high = np.array([120, 255,180]) # Rango alto de tono (rojo)
    mask = cv2.inRange(img_HSV, range_low, range_high)
    kernel = np.ones((2,2), np.uint8)
    _, img_bin = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
    img_er = cv2.erode(img_bin, kernel, iterations=3)
    kernel = np.ones((5,5), np.uint8)
    duckie = cv2.dilate(img_er, kernel, iterations=5)
    _, _, _, centroids = cv2.connectedComponentsWithStats(duckie, connectivity=8)
    return centroids[1]
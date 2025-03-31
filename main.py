"""
A machine learning algorithm to train for a Hide and Seek game.


Basic Flow:
* Create team of enemies and player.
* Create play area (size and obstacles).
* Move enemies to maximize objectives (hit player without being hit).
* Store data and use reinforced-learning.
"""

# Standard libraries
import math
import random
import time
from typing import List,Set,Tuple
from dataclasses import dataclass

# Local libraries

# 3rd-party Libraries
import pygame #this needs to be coded in such a way that you can extract pygame and still run the simulation.
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

#GLOBAL STATE VARIABLES
WIDTH = 800
HEIGHT = 600

OFFLINE_LEARNING_STEPS = 1000 #how long to run the simulation before starting to apply the training results

SAFE_REGION = [(50,50), (600, 300)] #[Xmin, Ymin],[Xmax,Ymax] => don't use "PyGame.rect" so PyGame can be excluded and the simulation part still run.
OBSTACLES = [(300, 250, 80, 50), (400, 300, 120, 30), (200, 450, 80, 40)]
WAYPOINTS = [(100, 300), (400, 200), (750, 350), (400, 500), (100, 300)]
WAYPOINT_ACCEPTABLE_DISTANCE = 5

PLAYER_SPEED = .6 #frame rate dependent speed (should be real time for real applications)
AI_SPEED = .6

PLAYER_DISPLAY_SIZE = 10
VISION_CONE_RANGE = 150
VISION_CONE_ANGLE = 60
PLAYER_MAX_HEALTH = 3
PLAYER_ROTATION_RATE = 1
PLAYER_PERCEPTION_RANGE = 200 #the maximum range the player will try to rotate towards an AI Unit

ENEMY_UNIT_DISPLAY_SIZE = 5
ENEMY_TEAM_SIZE = 5
ENEMY_DMG_RANGE = 50

WEIGHT_MAX_PER_DISTANCE = 20 #the maximum reward weight awarded in relation to distance from player.
WEIGHT_MAX_PER_FAR_DISTANCE = -30 #the maximum penalty for getting too far.
DISTANCE_NEAR = 200 #the distance that is considered near for calculating reward.
DISTANCE_FAR = 400 #the distance that is considered far for calculating penalty.
WEIGHT_MAX_PER_ANGLE = 45 #the maximum reward weight awarded when moving behind player.
WEIGHT_VISION_CONE_HIT = -40 #the action penalty for being caught in the vision cone.
WEIGHT_BLOCKED_VIEW = 30 #the reward for having a physical barrier between unit and player.
WEIGHT_OBSTACLE_HIT = 100 #the penalty for moving into an obstacle.
WEIGHT_PLAYER_HIT = 50 #the reward for hitting the player and not getting hit.

VISUALIZATION_MODE = True
BACKGROUND = (255, 255, 255) #background color
RESET_COLOR = (0, 0, 0) # reset state color
PLAYER_COLOR = (255, 0, 0) # player color
CONE_COLOR = (0, 0, 255, 100)  # semi-transparent blue for vision cone
AI_COLOR = (0, 255, 0)  # AI Units
OBSTACLE_COLOR = (100, 100, 100)  # obstacles
HEALTH_COLOR = (255, 255, 0)  # health Text Color

@dataclass
class Unit:
    position: Tuple[float, float]
    direction: Tuple[float, float]

    @classmethod
    def create_n(cls, n: int)-> List["Unit"]:
        return [cls(position = (random.uniform(SAFE_REGION[0][0],SAFE_REGION[1][0]),
                                random.uniform(SAFE_REGION[0][1],SAFE_REGION[1][1])),
                                direction = (random.uniform(-1,1),random.uniform(-1,1)))
                                for _ in range(n)]
    @classmethod
    def return_to_safe_zone(cls)-> Tuple[Tuple[float,float], Tuple[float,float]]:
        return (random.uniform(SAFE_REGION[0][0],SAFE_REGION[1][0]), random.uniform(SAFE_REGION[0][1],SAFE_REGION[1][1])), (random.uniform(-1,1),random.uniform(-1,1))
        
    
@dataclass
class MemoryItem:
    positionX: float
    positionY: float
    distance_to_player: float
    is_view_blocked_by_obstacle: float
    hit_obstacle: float
    is_within_vision_cone: float
    angle_relative_to_player: float

    def __str__(self):
        return f"Position:{self.positions} Distance: {self.distance_to_player} View Bocked: {self.is_view_blocked_by_obstacle}"

    def calculate_reward(self) -> int:
        
        positioning_reward = max(WEIGHT_MAX_PER_ANGLE - abs(self.angle_relative_to_player), 0)
        
        if self.distance_to_player <= DISTANCE_NEAR:
            distance_reward = min(WEIGHT_MAX_PER_DISTANCE, max(0,WEIGHT_MAX_PER_DISTANCE-(self.distance_to_player)/10))
        elif self.distance_to_player >= DISTANCE_FAR:
            distance_reward = WEIGHT_MAX_PER_FAR_DISTANCE
        else:
            distance_reward = 0
        
        if (self.is_view_blocked_by_obstacle):
            blocked_reward = WEIGHT_BLOCKED_VIEW
        else:
            blocked_reward = 0
        
        if (self.is_within_vision_cone):
            vision_cone_reward = WEIGHT_VISION_CONE_HIT
        else:
            vision_cone_reward = 0
        
        if (self.hit_obstacle):
            hit_obstacle_reward = WEIGHT_OBSTACLE_HIT
        else:
            hit_obstacle_reward = 0

        return positioning_reward + distance_reward + blocked_reward + vision_cone_reward + hit_obstacle_reward



class CustomRLAgentEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # A continuous space, the angle represented in a -1 to 1 space.
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)  # direction_taken (-1 to 1) => can transform into degrees later

        # Define observation space
        self.observation_space = spaces.Box(low=np.array([-np.inf, -np.inf, 0, 0, 0, 0, -np.inf]),  # lower bounds
                                            high=np.array([np.inf, np.inf, np.inf, 1, 1, 1, np.inf]),  # upper bounds
                                            dtype=np.float32)

        # Initial state variables (you would initialize this based on your environment)
        self.state = np.zeros(7, dtype=np.float32)  # Placeholder state
        self.done = False
    
    def step(self, action):
        #takes an action and returns the next state, reward, termination, info(optional), truncated?
        observation = None #the next state after takin the action
        reward = 0 #the reward for taking the action
        terminated = False #episode has ended (I don't think this simulation ends ever)
        truncated = None
        info = None #optional additional information

        return observation, reward, terminated, info    
    
    def render(self): #rendering is done outside of this class, so it doesn't matter
        pass  
    
    def reset(self): #doesn't really matter now, because I never reset the simulation
        self.state = np.zeros(7, dtype=np.float32)  # Placeholder state
        self.done = False

    #def close(self): #needs to be called to free-resouces
    #    pass

   

    
    def render(self, mode="Human"):
        pass

    def get_state(self):
        return np.random.rand(7).astype(np.float32)  # Replace with actual logic

    @staticmethod
    def calculate_reward(angle_relative_to_player, distance_to_player, is_view_blocked_by_obstacle,is_within_vision_cone,hit_obstacle ) -> int:
        positioning_reward = max(WEIGHT_MAX_PER_ANGLE - abs(angle_relative_to_player), 0)
        
        if distance_to_player <= DISTANCE_NEAR:
            distance_reward = min(WEIGHT_MAX_PER_DISTANCE, max(0,WEIGHT_MAX_PER_DISTANCE-(distance_to_player)/10))
        elif distance_to_player >= DISTANCE_FAR:
            distance_reward = WEIGHT_MAX_PER_FAR_DISTANCE
        else:
            distance_reward = 0
        
        if (is_view_blocked_by_obstacle):
            blocked_reward = WEIGHT_BLOCKED_VIEW
        else:
            blocked_reward = 0
        
        if (is_within_vision_cone):
            vision_cone_reward = WEIGHT_VISION_CONE_HIT
        else:
            vision_cone_reward = 0
        
        if (hit_obstacle):
            hit_obstacle_reward = WEIGHT_OBSTACLE_HIT
        else:
            hit_obstacle_reward = 0

        return positioning_reward + distance_reward + blocked_reward + vision_cone_reward + hit_obstacle_reward
"""
class PPO(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPO, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
"""

class Simulation():
    def __init__(self):
        self.offline_training_state = True
        self.iteration_index = 0

        self.player = Unit(WAYPOINTS[0],(0,1))
        self.player_health = PLAYER_MAX_HEALTH
        self.waypoint_index = 0

        self.AI_team = Unit.create_n(ENEMY_TEAM_SIZE)

        # Graphics Initialization
        if (VISUALIZATION_MODE):
            self.vis = Visualization(WIDTH, HEIGHT)
        else:
            self.vis = None

        self.abs_rects: List[Tuple[float,float,float,float]] = []
        for obstacles in OBSTACLES:
            self.abs_rects.append(
                self.convert_relative_rect_to_absol_rect(
                    (obstacles[0],obstacles[1],obstacles[2],obstacles[3])
                ))
            
        self.state_li: List[List[float, float, float, float, float, float, float]] = []
        self.action_li: List[float] = []
        self.reward_li: List[int] = []

        self.states: np.array = np.array([])
        self.actions: np.array = np.array([])
        self.rewards: np.array = np.array([])

        self.debug_message = ""

        self.start_simulation()

    def start_simulation(self):
              
        #move AI, towards player if possible
        #check if within cone, if it does, remove from to safe position
        #check if within damaging distance, if it does, deal 1 damage to player
        
        sim_running = True
        while (sim_running):

            self.iteration_index += 1
            if (self.iteration_index > OFFLINE_LEARNING_STEPS and self.offline_training_state == True):
                self.offline_training_state = False
               
                # initialize the environment, model
                self.env = CustomRLAgentEnv() #the environment setup, includes the state, action and rewards
                self.ppo_model = PPO("MlpPolicy", env=self.env, verbose=1) #initializing the mode
                self.ppo_model.learn(total_timesteps=10000)
                
                #self.ppo_model.save("ppo_hideNseek")
                #del ppo_model
                #self.ppo_model.load("ppo_hideNseek")

                self.obs = self.env.reset()

                


        
            # move player to next waypoint index, pass to next waypoint if near end of waypoint
            self.player.position, Move_direction = self.move_to_point(self.player.position, WAYPOINTS[(self.waypoint_index + 1) % len(WAYPOINTS)], PLAYER_SPEED)
            if (self.calculate_distance(self.player.position,WAYPOINTS[(self.waypoint_index + 1) % len(WAYPOINTS)]) <= WAYPOINT_ACCEPTABLE_DISTANCE):
                self.waypoint_index = (self.waypoint_index + 1) % len(WAYPOINTS)
            
            AI_direction, is_AI_close = self.rotate_player_towards_ai(self.player.position, self.AI_team, PLAYER_ROTATION_RATE, self.player.direction)
            # determine if you use the movement direction or turn to AI direction
            if (is_AI_close):
                self.player.direction = AI_direction
            else:
                self.player.direction = Move_direction

            #move AI, check for collisions, check for damaging player, check for being damage by the player
            for unit in self.AI_team:
                if self.offline_training_state:
                    unit.position, unit.direction = self.move_AI_towards_player(unit, self.player, self.abs_rects)
                else:
                    unit.position, unit.direction = self.move_AI_towards_player(unit, self.player, self.abs_rects)
                    self.ppo_model.learn(total_timesteps=50000)
                    




            if (VISUALIZATION_MODE):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sim_running = False
                        

                self.vis.screen.fill(BACKGROUND)
                pygame.draw.circle(self.vis.screen, PLAYER_COLOR, (self.player.position[0], self.player.position[1]), PLAYER_DISPLAY_SIZE)

                self.vis.draw_vision_cone(self.vis.screen, self.player.position, self.player.direction, VISION_CONE_RANGE, VISION_CONE_ANGLE)

                for ai_unit in self.AI_team:
                    pygame.draw.circle(self.vis.screen, AI_COLOR, (int(ai_unit.position[0]), int(ai_unit.position[1])), ENEMY_UNIT_DISPLAY_SIZE)
                
                for obstacle in OBSTACLES:
                    pygame.draw.rect(self.vis.screen, OBSTACLE_COLOR, pygame.Rect(obstacle[0], obstacle[1], obstacle[2], obstacle[3]))

                for waypoint in WAYPOINTS:
                    pygame.draw.circle(self.vis.screen, OBSTACLE_COLOR, (waypoint[0], waypoint[1]), 5)

                #health_text = self.vis.health_font.render(f"Health: {self.player_health}", True, HEALTH_COLOR)
                health_text = self.vis.health_font.render(f"Interation: {self.iteration_index}", True, HEALTH_COLOR)
                self.vis.screen.blit(health_text, (10,10))
                

                pygame.display.flip()
                pygame.time.Clock().tick(60)
        
        pygame.quit()

    def add_action_result_to_memory(self, position: Tuple[float,float], relative_angle: float, distance: float, view_blocked: bool, hit_obstacle: bool, is_within_vision_cone:bool, direction: float):
        
        new_state = [position[0],position[1], relative_angle, distance, float(view_blocked), float(hit_obstacle), float(is_within_vision_cone)]
        
        self.state_li.append(new_state)
        self.action_li.append(direction)
        self.reward_li.append(CustomRLAgentEnv.calculate_reward(angle_relative_to_player=relative_angle, distance_to_player=distance, is_view_blocked_by_obstacle=view_blocked, is_within_vision_cone=is_within_vision_cone, hit_obstacle=hit_obstacle))

        self.states = np.array(self.state_li, dtype=np.float32)
        self.actions = np.array(self.action_li, dtype=np.float32)
        self.rewards = np.array(self.reward_li)
  
    def reset_player(self):
        self.player = Unit(WAYPOINTS[0],(0,1))
        self.player_health = PLAYER_MAX_HEALTH

    @staticmethod
    def vector_to_angle_rad(Direction: Tuple[float, float])->float:
        return math.atan2(Direction[1], Direction[0])
    
    @staticmethod
    def angle_rad_to_vector(Angle: float) -> Tuple[float, float]:
        return math.cos(Angle), math.sin(Angle)
    
    @staticmethod 
    def points_to_angle_rad(origin: Tuple[float,float], target: Tuple[float,float])-> float:
        return math.atan2(target[1]-origin[1], target[0]-origin[0])
            
    @staticmethod
    def calculate_distance(pos1:Tuple[float, float], pos2:Tuple[float, float])-> float:
            d = math.sqrt((pos1[0]-pos2[0]) ** 2 + (pos1[1]-pos2[1])** 2)
            return float(d)
    
    @staticmethod
    def check_is_within_sector_angle(pos1:Tuple[float,float], pos2:Tuple[float,float], direction:Tuple[float, float]) -> bool: #outside cone sector
        dist = Simulation.calculate_distance(pos1, pos2)
        angle = Simulation.vector_to_angle_rad(direction)

        if dist > 0:
            target_angle = Simulation.points_to_angle_rad(pos1, pos2)
            delta_angle = abs(angle - target_angle)
            delta_angle = (delta_angle + math.pi) % (2 * math.pi) - math.pi #-pi to pi normalization
            return (delta_angle <= math.radians(VISION_CONE_ANGLE/2))
        else:
            return True #points are the same
        
    @staticmethod
    def is_in_vision_cone(player_position: Tuple[float,float], player_direction: Tuple[float,float], ai_position: Tuple[float,float]) -> Tuple[bool, bool]: #within range, within cone angle
        
        within_range = (Simulation.calculate_distance(player_position,ai_position) < VISION_CONE_RANGE) #too far
        within_cone_angle = Simulation.check_is_within_sector_angle(player_position,ai_position,player_direction)

        return within_range, within_cone_angle
        
    @staticmethod
    def move_to_point(initial_pos:Tuple[float,float], target_position:Tuple[float,float], speed:float) -> Tuple[Tuple[float,float], Tuple[float,float]]:
        distance = Simulation.calculate_distance(initial_pos,target_position)
        dirX, dirY = target_position[0]-initial_pos[0], target_position[1]-initial_pos[1]
        dirX, dirY = dirX/distance, dirY/distance

        posX, posY = initial_pos[0] + dirX * speed, initial_pos[1] + dirY * speed
        return ((posX, posY), (dirX, dirY))
    
    @staticmethod
    def rotate_player_towards_ai(player_pos: Tuple[float, float], ai_team: List[Unit], rotation_rate: float, player_direction: Tuple[float, float]) -> Tuple[Tuple[float, float], bool]:
    #Rotate the player to always face the closest AI, with max rotation rate.
        closest_ai = None
        closest_distance = float('inf')

        # Find the closest AI
        for ai in ai_team:
            ai_x, ai_y = ai.position[0], ai.position[1]
            dist = math.sqrt((ai_x - player_pos[0]) ** 2 + (ai_y - player_pos[1]) ** 2)
            if dist < closest_distance:
                closest_distance = dist
                closest_ai = ai
        
        if closest_distance > PLAYER_PERCEPTION_RANGE:
            return player_direction, False

        
        if closest_ai:
            # Calculate direction to closest AI
            ai_x, ai_y = closest_ai.position[0], closest_ai.position[1]
            dx = (ai_x - player_pos[0]) 
            dy = (ai_y - player_pos[1])

            # Normalize direction
            length = math.sqrt(dx ** 2 + dy ** 2)
            if length > 0:
                dx /= length
                dy /= length

            # Target Angle Player should turn to
            target_angle = math.atan2(dy, dx)
            
            # Calculate the angle between current direction and target direction
            player_angle = math.atan2(player_direction[1],player_direction[0])
            #angle_diff = (player_angle - target_angle) % (2 * math.pi)
            angle_diff = (target_angle - player_angle)
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi  # Get the shortest angle

            # Limit the angle change to ROTATION_RATE
            if abs(angle_diff) > math.radians(rotation_rate):
                angle_diff = math.radians(rotation_rate) * (1 if angle_diff > 0 else -1)

            # Update the player's direction
            player_angle += angle_diff
            
            # Convert back to dx, dy
            dx = math.cos(player_angle)
            dy = math.sin(player_angle)

            new_direction = (dx,dy)

            return new_direction, True
        return player_direction, False
    
    @staticmethod 
    def is_acceptable_range(position:Tuple[float,float], target_position:Tuple[float,float], acceptable_radius: float)-> bool:
        return (Simulation.calculate_distance(position,target_position) <= acceptable_radius)
    
    
    @staticmethod
    def convert_relative_rect_to_absol_rect(rect: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        x, y, width, height = rect
        return x, y, x+width, y+height
    
    @staticmethod
    def is_point_inside_rectangle(point: Tuple[float,float], rect: Tuple[float, float, float, float]) -> bool:
        x, y = point
        rx_min, ry_min, rx_max, ry_max = rect #rectangle must have being converted to absolute before hand
        return rx_min <= x <= rx_max and ry_min <= y <= ry_max
    
    @staticmethod
    def check_all_point_in_rectangles(point: Tuple, obstacles: List[Tuple[float, float, float, float]])-> bool:
        for obstacle in obstacles:
            if Simulation.is_point_inside_rectangle(point, obstacle):
                return True
                break
        return False
    
    
    def move_AI_towards_player(self, unit:Unit, player:Unit, obstacles:List[Tuple[float,float,float,float]])-> Tuple[Tuple[float,float], Tuple[float,float]]:
        #is player looking at me with no blocking? move perpendicular to player (any)
        #try to move towards player => got blocked? go back with slight change in angle

        is_player_cone_in_range, is_player_facing_unit = Simulation.is_in_vision_cone(player.position,player.direction,unit.position)
        is_player_view_blocked = Simulation.does_line_intersect_any_rectangle(unit.position,player.position, obstacles)
        is_player_within_perception_range = (Simulation.calculate_distance(unit.position, player.position) < PLAYER_PERCEPTION_RANGE)
                                             
        if (is_player_cone_in_range and is_player_facing_unit and not is_player_view_blocked): #you got hit, go back to safe space
            unit.position, unit.direction = unit.return_to_safe_zone()
            self.add_action_result_to_memory(unit.position, Simulation.angle_between_vectors(unit.direction,player.direction), Simulation.calculate_distance(unit.position, player.position), is_player_view_blocked, False, is_player_cone_in_range,unit.direction)
            return unit.position, unit.direction

        if (is_player_view_blocked) or not (is_player_within_perception_range): #(either you are blocked by an obstacle, or player is not facing you and outside of cone range
            try_position, try_direction = Simulation.move_to_point(unit.position, player.position, AI_SPEED)
        else:
            try_position, try_direction = Simulation.move_to_point(unit.position, player.position, AI_SPEED)
            perpendicular = Simulation.perpendicular_counterclockwise(try_direction)
            try_position = (unit.position[0] + perpendicular[0] * AI_SPEED * 100.0, unit.position[1] + perpendicular[1] * AI_SPEED * 100.0)
            try_position, try_direction = Simulation.move_to_point(unit.position, try_position, AI_SPEED) #####<<ERRO AQUI

        if (Simulation.check_all_point_in_rectangles(try_position,obstacles)):
            inverted = Simulation.invert_and_deviate(unit.direction, 45) #try to go in opposite direction with some randomization
            try_position = (unit.position[0] + inverted[0] * AI_SPEED * 100.0, unit.position[1] + inverted[1] * AI_SPEED * 100.0)
            try_position, try_direction = Simulation.move_to_point(unit.position, try_position, AI_SPEED)
            
            if (Simulation.check_all_point_in_rectangles(try_position,obstacles)):
                is_player_cone_in_range, is_player_facing_unit = Simulation.is_in_vision_cone(try_position,player.direction,unit.position)
                is_player_view_blocked = Simulation.does_line_intersect_any_rectangle(try_position,player.position, obstacles)
                is_player_within_perception_range = (Simulation.calculate_distance(try_position, player.position) < PLAYER_PERCEPTION_RANGE)

                self.add_action_result_to_memory(try_position, Simulation.angle_between_vectors(try_direction,player.direction), Simulation.calculate_distance(try_position, player.position), is_player_view_blocked, True, is_player_cone_in_range,try_direction)
                return unit.position, unit.direction #didn't find way to move, don't keep trying to move and stall the simulation
            else:
                is_player_cone_in_range, is_player_facing_unit = Simulation.is_in_vision_cone(try_position,player.direction,unit.position)
                is_player_view_blocked = Simulation.does_line_intersect_any_rectangle(try_position,player.position, obstacles)
                is_player_within_perception_range = (Simulation.calculate_distance(try_position, player.position) < PLAYER_PERCEPTION_RANGE)

                self.add_action_result_to_memory(try_position, Simulation.angle_between_vectors(try_direction,player.direction), Simulation.calculate_distance(try_position, player.position), is_player_view_blocked, False, is_player_cone_in_range,try_direction)
                return try_position, try_direction
        else:
            is_player_cone_in_range, is_player_facing_unit = Simulation.is_in_vision_cone(try_position,player.direction,try_direction)
            is_player_view_blocked = Simulation.does_line_intersect_any_rectangle(try_position,player.position, obstacles)
            is_player_within_perception_range = (Simulation.calculate_distance(try_position, player.position) < PLAYER_PERCEPTION_RANGE)
            self.add_action_result_to_memory(try_position, Simulation.angle_between_vectors(try_direction,player.direction), Simulation.calculate_distance(try_position, player.position), is_player_view_blocked, False, is_player_cone_in_range,try_direction)
            return try_position, try_direction     
        

    @staticmethod
    def invert_and_deviate(direction:Tuple[float,float], max_deviation:float) -> Tuple[float, float]:
    # Convert degrees to radians
        deviation = math.radians(max_deviation)
        deviation = random.uniform(-max_deviation,max_deviation)

        # Invert the vector
        x, y = -direction[0], -direction[1]

        # Apply rotation matrix
        x_rot = x * math.cos(deviation) - y * math.sin(deviation)
        y_rot = x * math.sin(deviation) + y * math.cos(deviation)

        return (x_rot, y_rot)        

    def angle_between_vectors(v1, v2):
        """Returns the angle between two 2D vectors in degrees."""
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]  # A · B
        magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)  # |A|
        magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)  # |B|

        if magnitude_v1 == 0 or magnitude_v2 == 0:
            return 0  # Avoid division by zero if a vector is (0,0)

        cos_theta = dot_product / (magnitude_v1 * magnitude_v2)  # Normalize
        cos_theta = max(-1, min(1, cos_theta))  # Clamp to avoid precision errors

        angle_radians = math.acos(cos_theta)  # Get angle in radians
        return math.degrees(angle_radians)  # Convert to degrees
    
    @staticmethod
    def perpendicular_clockwise(v: Tuple[float, float]) -> Tuple[float, float]:
        """Returns the perpendicular vector in the clockwise direction."""
        x, y = v
        return (y, -x)
    
    @staticmethod
    def perpendicular_counterclockwise(v: Tuple[float, float]) -> Tuple[float, float]:
        """Returns the perpendicular vector in the counterclockwise direction."""
        x, y = v
        return (-y, x)

       
    def line_intersects_line(p1: Tuple[float,float], p2: Tuple[float,float], q1: Tuple[float,float], q2: Tuple[float,float]):
        """Check if line segment (p1-p2) intersects with (q1-q2) using cross products."""
        
        def cross_product(v1, v2):
            return v1[0] * v2[1] - v1[1] * v2[0]

        def subtract(v1:Tuple[float,float], v2:Tuple[float,float]):
            return (v1[0] - v2[0], v1[1] - v2[1])

        r = subtract(p2, p1)
        s = subtract(q2, q1)

        rxs = cross_product(r, s)
        qpxr = cross_product(subtract(q1, p1), r)

        if rxs == 0:
            return False  # Parallel or collinear (no intersection)

        t = cross_product(subtract(q1, p1), s) / rxs
        u = qpxr / rxs

        return 0 <= t <= 1 and 0 <= u <= 1  # True if intersection is within both segments

    def line_intersects_rectangle(p1:Tuple[float,float], p2:Tuple[float,float], rect: Tuple[float, float, float, float]):
        """Check if a line segment (p1-p2) intersects any edge of the rectangle."""
        
        x_min, y_min, x_max, y_max = rect

        edges = [
        ((x_min, y_min), (x_max, y_min)),  # Bottom edge
        ((x_max, y_min), (x_max, y_max)),  # Right edge
        ((x_max, y_max), (x_min, y_max)),  # Top edge
        ((x_min, y_max), (x_min, y_min)),  # Left edge
        ]

        for edge in edges:
            if Simulation.line_intersects_line(p1, p2, edge[0], edge[1]):
                return True  # Early exit if any intersection is found

        return False

    @staticmethod
    def does_line_intersect_any_rectangle(p1, p2, rectangles: List[Tuple[float, float, float, float]]) -> bool:
        """Check if a line segment (p1-p2) intersects any rectangle in the list."""
        for rect in rectangles:
            if Simulation.line_intersects_rectangle(p1, p2, rect):
                return True
        return False
    

class Visualization():
    #Initialize PyGame visuals. This class should be built in a way that the underlying data does not depend on it (as in, run without visualization is ok)   
    def __init__(self, width: float=800, height: float=600):
        self.WIDTH = width
        self.HEIGHT = height
        pygame.font.init()
        self.health_font = pygame.font.SysFont('Arial', 24)

        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH,self.HEIGHT))
        pygame.display.set_caption("2D AI Avoidance Training")

    def draw_vision_cone(self, surface, pos, direction, vision_range, vision_angle):
        #Draws a vision cone originating from pos, facing direction.
        half_angle = math.radians(vision_angle / 2)

        # Convert direction to unit vector
        dx, dy = direction
        dir_length = math.sqrt(dx**2 + dy**2)
        if dir_length > 0:
            dx /= dir_length
            dy /= dir_length

        # Calculate left and right edges of cone
        angle_player = math.atan2(dy,dx) #fix cone direction
        angle_left = angle_player + half_angle
        angle_right = angle_player - half_angle

        left_x = pos[0] + vision_range * math.cos(angle_left)
        left_y = pos[1] + vision_range * math.sin(angle_left)

        right_x = pos[0] + vision_range * math.cos(angle_right)
        right_y = pos[1] + vision_range * math.sin(angle_right)

        # Draw the vision cone as a triangle
        pygame.draw.polygon(surface, CONE_COLOR, [(pos[0], pos[1]), (left_x, left_y), (right_x, right_y)])

    def display_message(self, message):
        #Display a message on the screen.
        text_surface = pygame.font.SysFont('Arial', 48).render(message, True, (255, 255, 255))
        self.screen.blit(text_surface, (WIDTH // 2 - text_surface.get_width() // 2, HEIGHT // 2 - text_surface.get_height() // 2))
        pygame.display.flip()
        time.sleep(2)  # Pause for 2 seconds

    
if __name__=="__main__":
    Simulation()    

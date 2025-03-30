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

#GLOBAL STATE VARIABLES
WIDTH = 800
HEIGHT = 600

OFFLINE_LEARNING_STEPS = 100 #how long to run the simulation before starting to apply the training results

SAFE_REGION = [(50,50), (600, 300)] #[Xmin, Ymin],[Xmax,Ymax] => don't use "PyGame.rect" so PyGame can be excluded and the simulation part still run.
OBSTACLES = [(300, 250, 100, 50), (500, 350, 120, 60), (200, 450, 80, 40)]
WAYPOINTS = ([(100, 300), (400, 200), (700, 300), (400, 500), (100, 300)])
WAYPOINT_ACCEPTABLE_DISTANCE = 5

PLAYER_SPEED = 0.3 #frame rate dependent speed (should be real time for real applications)
AI_SPEED = 0.3

PLAYER_DISPLAY_SIZE = 10
VISION_CONE_RANGE = 150
VISION_CONE_ANGLE = 90
PLAYER_MAX_HEALTH = 3
PLAYER_ROTATION_RATE = 15

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
    direction: Tuple [float, float]

    @classmethod
    def create_n(cls, n: int)-> List["Unit"]:
        return [cls(position = (random.uniform(SAFE_REGION[0],SAFE_REGION[2]),
                                random.uniform(SAFE_REGION[1],SAFE_REGION[3])),
                                direction = (random.uniform(-1,1),random.uniform(-1,1)))
                                for _ in range(n)]
    
@dataclass
class MemoryItem:
    positions: Tuple[float,float] #X, Y
    distance_to_player: float
    is_view_blocked_by_obstacle: bool
    hit_obstacle: bool
    is_within_vision_cone: bool
    angle_relative_to_player: float
    
    direction_taken: float #the movement action took

    reward: int

    def calculate_reward(self):
        
        positioning_reward = max(abs(WEIGHT_MAX_PER_ANGLE - self.angle_relative_to_player), 0)
        
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

        self.reward = positioning_reward + distance_reward + blocked_reward + vision_cone_reward + hit_obstacle_reward


        
        
        pass



    

class Simulation():
    def __init__(self):
        self.player = Unit(WAYPOINTS[0],(0,1))
        self.player_health = PLAYER_MAX_HEALTH

        self.AI_team = Unit.create_n(ENEMY_TEAM_SIZE)

        # Graphics Initialization
        if (VISUALIZATION_MODE):
            self.vis = Visualization(WIDTH, HEIGHT)
        else:
            self.vis = None


        self.start_simulation()



        

    
    
    def start_simulation():
        
        #move player, recalculate now waypoint in necessary
        #rotate player toward nearest AI
        
        #move AI, towards player if possible
        #check if within cone, if it does, remove from to safe position
        #check if within damaging distance, if it does, deal 1 damage to player


        
        pass


            

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
        return math.atan2(target[0]-origin[0], target[1]-origin[1])
            
    @staticmethod
    def calculate_distance(pos1:Tuple[float, float], pos2:Tuple[float, float])-> float:
            d = math.sqrt((pos1[0]-pos2[0]) ** 2 + (pos1[1]-pos2[1])** 2)
            return float(d)
    
    @staticmethod
    def check_is_within_sector_angle(pos1:Tuple[float,float], pos2:Tuple[float,float], direction:Tuple[float, float]) -> bool:
        dist = Simulation.calculate_distance(pos1, pos2)
        angle = Simulation.vector_to_angle_rad(direction)

        if dist > 0:
            target_angle = Simulation.points_to_angle_rad(pos1, pos2)
            delta_angle = abs(angle - target_angle)
            return (delta_angle <= math.radians(VISION_CONE_ANGLE/2))
        else:
            return True #points are the same
        
    @staticmethod
    def is_in_vision_cone(player_position: Tuple[float,float], player_direction: Tuple[float,float], ai_position: Tuple[float,float]) -> bool:
        if (Simulation.calculate_distance(player_position,ai_position) > VISION_CONE_RANGE): #too far
            return False
        else:
            return Simulation.check_sector_angle(player_position,ai_position,player_direction)
        
    @staticmethod
    def move_to_point(initial_pos:Tuple[float,float], target_position:Tuple[float,float], speed:float) -> Tuple [(float,float),(float,float)]:
        distance = Simulation.calculate_distance(initial_pos,target_position)
        dirX, dirY = target_position[0]-initial_pos[0], target_position[1]-initial_pos[1]
        dirX, dirY = dirX/distance, dirY/distance

        posX, posY = initial_pos[0] + dirX * speed, initial_pos[1] + dirY
        return [(posX, posY), (dirX, dirY)]
    
    @staticmethod 
    def is_acceptable_range(position:Tuple[float,float], target_position:Tuple[float,float], acceptable_radius: float)-> bool:
        return (Simulation.calculate_distance(position,target_position) <= acceptable_radius)
    
    @staticmethod
    def move_towards_player():
        pass

    
    
    



    

class AIUnit():
    def __init__(self, speed: float = 0.3, health: int = 3, position_X:float = 0.0, position_Y:float = 0.0, direction=Tuple[float, float]):
        self.speed = speed
        self.health = health
        self.position = (position_X, position_Y)
        self.direction = direction

        self.IsDead = False

    def deal_damage(self, dmg_amount: int = 1):
        self.health -= dmg_amount
        self.check_is_dead()
    
    def check_is_dead(self)-> bool:
        if (self.health <= 0):
            self.IsDead = True
            return True
        else:
            return False


class AITeam():
    def __init__(self):
        self.team: List[AIUnit] = []
    
    def __len__(self):
        return len(self.team)

    def add_to_team(self, new_team_member: AIUnit):
        self.team.append(new_team_member)

class Player():
    def __init__(self, player_size: float = 20, player_speed: float = 0.3, health: int =3, vision_angle: float = 60, vision_range: float = 150, rotation_rate: float = 5):
        self.size = player_size
        self.speed = player_speed
        self.max_health = health
        self.health = health
        self.vision_angle = vision_angle
        self.vision_range = vision_range
        self.rotation_rate = rotation_rate
        self.position: Tuple[float,float] = (0.0, 0.0)

        self.direction : Tuple[float,float] = (0,0) # where player is pointed to (angle)
        self.waypoints = [(0,0)]
        self.waypoint_index = 0
        self.waypoint_lastindex = len(self.waypoints) - 1

    def set_waypoints(self, waypoints: List[Tuple[int, int]]):
        self.waypoints = waypoints
        self.total_waypoints = len(self.waypoints)
        self.position = waypoints[0]
        self.waypoint_lastindex = len(self.waypoints) - 1

    def reset_player(self, new_waypoints: List[Tuple[int, int]]):
        self.health = self.max_health
        self.waypoint_index = 0
        self.position = self.waypoints[0]

        if (new_waypoints):
            self.waypoints = new_waypoints

class Game():
        def __init__(self):
            
            self.obstacles : List[Tuple[int, int, int, int]]

        def add_geometry(self, obstacles: List[Tuple[int, int, int, int]]):
            self.obstacles = obstacles
       
        def move_ai_units(self, ai_team: AITeam, player_pos: Tuple[int, int], player_dir: Tuple[float, float], vision_range: int, vision_angle: float):

            """Moves AI units, and avoids vision cone."""
            for ai in ai_team.team:
                # Determine direction to move towards player
                target_x, target_y = player_pos
                ai_dir_x = target_x - ai.position[0]
                ai_dir_y = target_y - ai.position[1]
                ai_dist = math.sqrt(ai_dir_x ** 2 + ai_dir_y ** 2)

                if ai_dist > 0:
                    ai_dir_x /= ai_dist
                    ai_dir_y /= ai_dist
                else:
                    ai_dir_x = 0
                    ai_dir_y = 0

                # Check if AI is within the vision cone, and avoid it
                if self.is_ai_in_vision_cone(ai.position, player_pos, player_dir, vision_range, vision_angle):
                    # If AI is in vision cone, it 'dies', that is, move to the safe region
                    random_x = random.randint(SAFE_REGION[0][0], SAFE_REGION[1][0])
                    random_y = random.randint(SAFE_REGION[0][1], SAFE_REGION[1][1])
                    ai.position = (random_x, random_y)
                else:
                    # If not in vision cone, move towards the player
                    ai.direction = (ai_dir_x, ai_dir_y)

                # Normalize direction vector
                ai_length = math.sqrt(ai.direction[0]**2 + ai.direction[1]**2)
                if ai_length > 0:
                    ai_dx = ai.direction[0] / ai_length
                    ai_dy = ai.direction[1] / ai_length
                    ai.direction = (ai_dx, ai_dy)

                # Move AI unit
                px = ai.position[0] + ai.direction[0] * ai.speed
                py = ai.position[1] + ai.direction[1] * ai.speed
                ai.position = (px,py)

                # Bounce off walls
                if ai.position[0] < 50 or ai.position[0] > WIDTH - 50:
                    ai.direction[0] *= -1
                if ai.position[1] < 50 or ai.position[1] > HEIGHT - 50:
                    ai.direction[1] *= -1
    
        def line_intersects_rect(self, p1, p2, rect):
            """Checks if a line segment (p1 to p2) intersects a rectangle."""
            collision = False
            try:
                re = [
                    ([rect[0], rect[1]], [rect[0]+rect[2], rect[1]]),
                    ([rect[0] + rect[2], rect[1]], [rect[0] + rect[2], rect[1] + rect[3]]),
                    ([rect[0] + rect[2], rect[1] + rect[3]], [rect[0], rect[1] + rect[3]]),
                    ([rect[0], rect[1] + rect[3]], [rect[0], rect[1]])
                ]
                
                def line_intersects_line(a1, a2, b1, b2):
                    """Checks if two line segments (a1-a2 and b1-b2) intersect."""
                    def ccw(A, B, C):
                        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
                    return ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2)

                for rect_line in re:
                    if line_intersects_line(p1, p2, rect_line[0], rect_line[1]):
                        return True
            except:
                return False
            return collision

        def is_ai_in_vision_cone(self, ai_pos, player_pos, direction, vision_range, vision_angle) -> bool:
            """Check if an AI unit is within the vision cone and not blocked by obstacles."""
            ax, ay = ai_pos
            px, py = player_pos
            dx, dy = direction

            # Distance check
            dist = math.sqrt((ax - px) ** 2 + (ay - py) ** 2)
            if dist > vision_range:
                return False  # Too far away

            # Angle check
            ai_dir_x, ai_dir_y = ax - px, ay - py
            ai_dist = math.sqrt(ai_dir_x ** 2 + ai_dir_y ** 2)
            if ai_dist == 0:
                return False  # AI is at the same position as player

            # Normalize AI direction
            ai_dir_x /= ai_dist
            ai_dir_y /= ai_dist

            # Normalize Player direction
            player_dist = math.sqrt(dx ** 2 + dy ** 2)
            if player_dist > 0:
                dx /= player_dist
                dy /= player_dist

            # Compute dot product
            dot_product = ai_dir_x * dx + ai_dir_y * dy
            dot_product = max(-1, min(1, dot_product))
            angle = math.degrees(math.acos(dot_product))

            if angle > (vision_angle / 2):
                return False  # AI is outside the vision cone

            # Check if vision is blocked
            for obstacle in self.obstacles:
                if self.line_intersects_rect(player_pos, ai_pos, obstacle):
                    return False  # AI is blocked
            return True
    
        def rotate_player_towards_ai(self, player_pos: Tuple[float, float], AI_team:AITeam, rotation_rate: float, player_direction: Tuple[float, float]):
            """Rotate the player to always face the closest AI, with max rotation rate."""
            closest_ai = None
            closest_distance = float('inf')

            # Find the closest AI
            for ai in AI_team.team:
                ai_x, ai_y = ai.position[0], ai.position[1]
                dist = math.sqrt((ai_x - player_pos[0]) ** 2 + (ai_y - player_pos[1]) ** 2)
                if dist < closest_distance:
                    closest_distance = dist
                    closest_ai = ai

            if closest_ai:
                # Calculate direction to closest AI
                ai_x, ai_y = closest_ai.position[0], closest_ai.position[1]
                dx = ai_x - player_pos[0]
                dy = ai_y - player_pos[1]

                # Normalize direction
                length = math.sqrt(dx ** 2 + dy ** 2)
                if length > 0:
                    dx /= length
                    dy /= length

                # Calculate current player facing direction (assumed to be right)
                current_angle = math.atan2(dy, dx)
                
                # Calculate the angle between current direction and target direction
                player_angle = math.atan2(player_direction[1],player_direction[0])
                angle_diff = (player_angle - current_angle) % (2 * math.pi)

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

                return (dx, dy)
            return (player_direction)
        
        def check_player_health(self, AI_team:AITeam, player_pos: Tuple[float,float]): #return TRUE does 1 damage.
            """Check if the player has lost health due to AI proximity."""
            for ai in AI_team.team:
                if (self.distance_check(player_pos, ai.position)<ENEMY_DMG_RANGE) and not self.line_intersects_rect(player_pos, ai.position, self.obstacles):
                    
                    return True  # player takes 1 damage
            return False
        
        def distance_check(self, pos1:Tuple[float, float], pos2:Tuple[float, float])-> float:
            d = math.sqrt((pos1[0]-pos2[0]) ** 2 + (pos1[1]-pos2[1])** 2)
            return float(d)
        
        def check_win_condition(self, AI_team: AITeam):
            """Check if all AI enemies are eliminated."""
            if len(AI_team.team) == 0:
                return True  # Player wins
            return False

        


class Visualization():
    """
    Initialize PyGame visuals. This class should be built in a way that the underlying data does not depend on it (as in, run without visualization is ok)
    """
    BACKGROUND = (255, 255, 255) #background color
    RESET_COLOR = (0, 0, 0) # reset state color
    PLAYER_COLOR = (255, 0, 0) # player color
    CONE_COLOR = (0, 0, 255, 100)  # semi-transparent blue for vision cone
    AI_COLOR = (0, 255, 0)  # AI Units
    OBSTACLE_COLOR = (100, 100, 100)  # obstacles
    HEALTH_COLOR = (255, 255, 0)  # health Text Color
    
    def __init__(self, width: float=800, height: float=600):
        self.WIDTH = width
        self.HEIGHT = height
        pygame.font.init()
        self.health_font = pygame.font.SysFont('Arial', 24)

        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH,self.HEIGHT))
        pygame.display.set_caption("2D AI Avoidance Training")

    def draw_vision_cone(self, surface, pos, direction, vision_range, vision_angle):
        """Draws a vision cone originating from pos, facing direction."""
        half_angle = math.radians(vision_angle / 2)

        # Convert direction to unit vector
        dx, dy = direction
        dir_length = math.sqrt(dx**2 + dy**2)
        if dir_length > 0:
            dx /= dir_length
            dy /= dir_length

        # Calculate left and right edges of cone
        angle_player = math.atan2(dy,dx)
        angle_left = angle_player + half_angle
        angle_right = angle_player - half_angle

        left_x = pos[0] + vision_range * math.cos(angle_left)
        left_y = pos[1] + vision_range * math.sin(angle_left)

        right_x = pos[0] + vision_range * math.cos(angle_right)
        right_y = pos[1] + vision_range * math.sin(angle_right)

        # Draw the vision cone as a triangle
        pygame.draw.polygon(surface, self.CONE_COLOR, [(pos[0], pos[1]), (left_x, left_y), (right_x, right_y)])

    def display_message(self, message):
        """Display a message on the screen."""
        text_surface = pygame.font.SysFont('Arial', 48).render(message, True, (255, 255, 255))
        self.screen.blit(text_surface, (WIDTH // 2 - text_surface.get_width() // 2, HEIGHT // 2 - text_surface.get_height() // 2))
        pygame.display.flip()
        time.sleep(2)  # Pause for 2 seconds



class main():
    def __init__(self):
        #Player Initialization
        self.player = Player()
        self.player.set_waypoints(WAYPOINTS)

        # AI team Initialization
        self.AI_team = AITeam()
        count = 0
        while (count  < ENEMY_TEAM_SIZE):
            new_enemy = AIUnit(speed=0.1,health=3,position_X=random.randint(SAFE_REGION[0][0], SAFE_REGION[1][0]),position_Y=random.randint(SAFE_REGION[0][1], SAFE_REGION[1][1]), direction=[random.choice([-1, 1]), random.choice([-1, 1])])
            self.AI_team.add_to_team(new_enemy)
            count += 1
              
        # Collision and movement handler initialization
        game = Game()
        game.add_geometry(OBSTACLES)


        # Graphics Initialization
        if (VISUALIZATION_MODE):
            self.vis = Visualization(WIDTH, HEIGHT)
        else:
            self.vis = None
        
 

            # Simulation Start
            
        

        running = True
        while running:

            # Rotate player towards closest AI with max rotation rate
            self.player.direction = game.rotate_player_towards_ai(self.player.position, self.AI_team, self.player.rotation_rate, self.player.direction)

            # Move player towards current waypoint
            target_x, target_y = self.player.waypoints[self.player.waypoint_index]
            dx, dy = target_x - self.player.position[0], target_y - self.player.position[1]
                    
            dist = math.sqrt(dx**2 + dy**2)

            #print(f"tx: {target_x} - ty: {target_y}")

            if dist > 0: #if the distance is than what it can move per iteration
                px = (self.player.position[0] + (self.player.speed * dx / dist))
                py = (self.player.position[1] + (self.player.speed * dy / dist))
                #print (f"P1: {px} - P2: {py}")
                self.player.position = (px,py)
                
            else:
                self.player.position = list(self.player.waypoints[self.player.waypoint_index])
                self.player.waypoint_index = (self.player.waypoint_index + 1) % len(self.player.waypoints)

            # Move AI units
            game.move_ai_units(self.AI_team, self.player.position, self.player.direction, self.player.vision_range, self.player.vision_angle)

            # Check player health
            if game.check_player_health(self.AI_team,self.player.position):
                # If health reaches 0, end game
                ai_units = []  # Reset AI units, they will be build again when game restarts.
                running = False
                break
                continue  # Skip this frame

            # Check if player has won (no more enemies)
            if game.check_win_condition(self.AI_team):
                # Reset AI units
                ai_units = [] # Reset AI units, they will be build again when game restarts.
                running = False
                break
                continue  # Skip this frame

            # Region Visualization


            if (VISUALIZATION_MODE):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                self.vis.screen.fill(self.vis.BACKGROUND)
                pygame.draw.circle(self.vis.screen, self.vis.PLAYER_COLOR, (self.player.position[0], self.player.position[1]), self.player.size)

                self.vis.draw_vision_cone(self.vis.screen, self.player.position, self.player.direction, self.player.vision_range, self.player.vision_angle)

                for ai_unit in self.AI_team.team:
                    pygame.draw.circle(self.vis.screen, self.vis.AI_COLOR, (int(ai_unit.position[0]), int(ai_unit.position[1])), ENEMY_TEAM_SIZE)

                for obstacle in OBSTACLES:
                    pygame.draw.rect(self.vis.screen, self.vis.OBSTACLE_COLOR, pygame.Rect(obstacle[0], obstacle[1], obstacle[2], obstacle[3]))

                for waypoint in WAYPOINTS:
                    pygame.draw.circle(self.vis.screen, self.vis.OBSTACLE_COLOR, (waypoint[0], waypoint[1]), 5)

                health_text = self.vis.health_font.render(f"Health: {self.player.health}", True, self.vis.HEALTH_COLOR)
                self.vis.screen.blit(health_text, (10,10))

                pygame.display.flip()
                pygame.time.Clock().tick(60)
        # endregion  

    #pygame.quit()

if __name__=="__main__":
    main()



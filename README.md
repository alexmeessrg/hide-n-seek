# Hide and Seek

A reinforced learning algorithm (PPO) for game AI with chase and avoid behavior. Uses Stable Diffusion 3 and PyGame (for displaying results).
"Player" moves on a pre-determined path and AI tries to get as close as it can. Player rotates toward nearby AI as to simulate actual player awareness of danger.

## Awarded Behavior: ##
* Get as close as possible to target.
* Get out of view of field of player.
* Get within hit range of player.
* Hide behind obstacles.

## Penalized Behavior: ##
* Getting with range and field of view of player.
* Bump into obstacles.


## Known bugs: ##
* Bad training for some initial conditions.
* Still needs work to speed up fitting.

## Features to add: ##
* Change training model to add multi-agent training.
  


## GLOBAL STATE VARIABLES ##
(and default values)

#screen size (simulation area)

**WIDTH** = 800

**HEIGHT** = 600

**OFFLINE_LEARNING_STEPS** = 5000 #how long to run the simulation before starting to apply the training results

**SAFE_REGION** = [(100,100), (600, 400)] #[Xmin, Ymin],[Xmax,Ymax] #where AI will spawn

**OBSTACLES** = [(300, 50, 80, 50), (400, 500, 120, 30), (200, 450, 80, 40)] #placeable obstacles

**WAYPOINTS** = [(150, 300), (400, 100), (750, 350), (400, 500), (300, 300)]

**WAYPOINT_ACCEPTABLE_DISTANCE** = 5 #how close you can get to a waypoint before looking for next.

**PLAYER_SPEED** = .4 #frame rate dependent speed (should be real time for real applications)

**AI_SPEED** = .7

**PLAYER_DISPLAY_SIZE** = 10

**VISION_CONE_RANGE** = 150

**VISION_CONE_ANGLE** = 60

**PLAYER_MAX_HEALTH** = 3 #not implemented

**PLAYER_ROTATION_RATE** = 0.5 #how fast the player will turn to face AI

**PLAYER_PERCEPTION_RANGE** = 200 #the maximum range the player will try to rotate towards an AI Unit

**UNIT_HIT_PLAYER_DISTANCE** = 50 #the range the unit will hit the player

**ENEMY_UNIT_DISPLAY_SIZE** = 5 

**ENEMY_TEAM_SIZE** = 1 #not implemented

**ENEMY_DMG_RANGE** = 50 #range AI will damage player


**WEIGHT_MAX_PER_DISTANCE** = 5 #the maximum reward weight awarded in relation to distance from player.

**WEIGHT_MAX_PER_FAR_DISTANCE** = -0 #the maximum penalty for getting too far (negative means less reinforced).

**DISTANCE_NEAR** = 50 #the distance that is considered near for calculating reward.

**DISTANCE_FAR** = 300 #the distance that is considered far for calculating penalty.

**WEIGHT_MAX_PER_ANGLE** = 2 #the maximum reward weight awarded when moving behind player.

**WEIGHT_VISION_CONE_HIT** = -1 #the action penalty for being caught in the vision cone (negative means less reinforced).

**WEIGHT_BLOCKED_VIEW** = 1 #the reward for having a physical barrier between unit and player.

**WEIGHT_OBSTACLE_HIT** = -2 #the penalty for moving into an obstacle (negative means less reinforced).

**WEIGHT_PLAYER_HIT** = 3 #the reward for hitting the player and not getting hit.


**VISUALIZATION_MODE** = True

**BACKGROUND** = (0, 0, 0) #background color

**RESET_COLOR** = (0, 0, 0) # reset state color

**PLAYER_COLOR** = (255, 0, 0) # player color

**CONE_COLOR** = (0, 0, 255, 100)  # semi-transparent blue for vision cone

**AI_COLOR** = (0, 255, 0)  # AI Units

**OBSTACLE_COLOR** = (100, 100, 100)  # obstacles

**HEALTH_COLOR** = (255, 255, 0)  # health Text Color


# Q's Vehicle Simulation
# By KyushikMin kyushikmin@gamil.com
# http://mmc.hanyang.ac.kr 

import random, sys, time, math, pygame
from pygame.locals import *

WINDOW_WIDTH = 720
WINDOW_HEIGHT = 720
HALF_WINDOW_WIDTH = int(WINDOW_WIDTH / 2)
HALF_WINDOW_HEIGHT = int(WINDOW_HEIGHT / 2)

# MAP_LIST = ['Straight_UD', 'Straight_LR', 'Cross']
MAP_LIST = ['Straight_UD']
DIRECTION_LIST = ['up', 'down', 'left', 'right']

RANDOM_VEHICLE_NUM = 2
RANDOM_OBSTACLE_NUM = 0
basic_reward = -0.00 
time_limit = 20

ROAD_WIDTH = 360 # Width of the road
LANE_WIDTH = ROAD_WIDTH/4 # Width of a lane
LINE_WIDTH = 8 # Width of a lines of lane and road

#				 R    G    B
WHITE        = (255, 255, 255)
GRAY 	     = (100, 100, 100)
BLACK		 = (  0,   0,   0)
RED 		 = (155,   0,   0)
LIGHT_RED    = (175,  20,  20)
GREEN		 = (  0, 155,   0)
LIGHT_GREEN  = ( 20, 175,  20)
BLUE 		 = (  0,   0, 155)
LIGHT_BLUE   = ( 20,  20, 175)
YELLOW 		 = (155, 155,   0)
LIGHT_YELLOW = (175, 175,  20)

Road_Edge_Color = RED
Road_Color = GRAY 
Center_Color = YELLOW 
Lane_Color = WHITE 
Text_Color = WHITE 
Background_Color = BLACK

HOST_IMAGE = pygame.image.load('./Wrapped_Game/Qarsim_file/Host.png')
REMOTE_IMAGES_DICT = {'Remote_1': pygame.image.load('./Wrapped_Game/Qarsim_file/Remote_1.png'), 
			   	 	  'Remote_2': pygame.image.load('./Wrapped_Game/Qarsim_file/Remote_2.png'),
			   	 	  'Remote_3': pygame.image.load('./Wrapped_Game/Qarsim_file/Remote_3.png'),
			   	 	  'Remote_4': pygame.image.load('./Wrapped_Game/Qarsim_file/Remote_4.png'),
			   	 	  'Remote_5': pygame.image.load('./Wrapped_Game/Qarsim_file/Remote_5.png'),
			   	 	  'Remote_6': pygame.image.load('./Wrapped_Game/Qarsim_file/Remote_6.png')}
OBSTACLE_IMAGE = pygame.image.load('./Wrapped_Game/Qarsim_file/Obstacle_1.png')
GOAL_IMAGE = pygame.image.load('./Wrapped_Game/Qarsim_file/goal.png')

UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'

pygame.init()

DISPLAYSURF = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

pygame.display.set_caption('Qar Sim')
pygame.display.set_icon(pygame.image.load('./Wrapped_Game/Qarsim_file/icon_resize2.png'))

BASIC_FONT = pygame.font.Font('freesansbold.ttf', 16)

# score = 0
class GameState:
	def __init__(self):
			self.Map = 'Straight_UD'

			self.Host_Obj = {'surface': UP,
							'x': HALF_WINDOW_WIDTH + LANE_WIDTH / 2.0,
							'y': WINDOW_HEIGHT - 50.0,
							'velocity': 10.0,
							'acc': 0.0,
							'steering': 0.0,
							'heading': 0.0,
							'width': 35,
							'length': 80,
							'image': HOST_IMAGE,
							'rect': 0}

			self.Host_Obj['rect'] = pygame.Rect( (self.Host_Obj['x'],
						     				      self.Host_Obj['y'],
						      			          self.Host_Obj['width'],
						      			          self.Host_Obj['length']))

			self.Remote_Obj_List = []

			self.score = 0

			for i in range(RANDOM_VEHICLE_NUM):
				Init_direction_list = ['up', 'down']
				Heading_list = [0, 180]
				self.Remote_Obj_List.append({'surface': random.choice(Init_direction_list),
											'x': random.randint(HALF_WINDOW_WIDTH - (ROAD_WIDTH / 4), HALF_WINDOW_WIDTH + (ROAD_WIDTH / 2) - 50),
											'y': random.randint(100, WINDOW_HEIGHT - 300),
											'velocity': random.randint(10, 40),
											'heading': 0,
											'image': REMOTE_IMAGES_DICT[random.choice(REMOTE_IMAGES_DICT.keys())],
											'width': 35,
											'length': 80,
											'rect': 0
											})

				self.Remote_Obj_List[i]['rect'] = pygame.Rect((self.Remote_Obj_List[i]['x'], 
															   self.Remote_Obj_List[i]['y'], 
															   self.Remote_Obj_List[i]['width'], 
															   self.Remote_Obj_List[i]['length']
															 ))

				if self.Remote_Obj_List[i]['surface'] == 'up':
					self.Remote_Obj_List[i]['heading'] = 0
				elif self.Remote_Obj_List[i]['surface'] == 'down':
					self.Remote_Obj_List[i]['heading'] = 180

				if self.Remote_Obj_List[i]['surface'] == 'down':
					self.Remote_Obj_List[i]['image'] = pygame.transform.rotate(self.Remote_Obj_List[i]['image'],self.Remote_Obj_List[i]['heading'])


			self.Obstacle_Obj_List = []

			for i in range(RANDOM_OBSTACLE_NUM):
				self.Obstacle_Obj_List.append({'x': random.randint(HALF_WINDOW_WIDTH, HALF_WINDOW_WIDTH + (ROAD_WIDTH / 2) - 50),
										 	   'y': random.randint(0 + 250, WINDOW_HEIGHT - 250),
									 		   'image': OBSTACLE_IMAGE,
									 		   'width': 30,
									 		   'length': 20,
									 		   'rect': 0})
				self.Obstacle_Obj_List[i]['rect'] = pygame.Rect((self.Obstacle_Obj_List[i]['x'],
									  					      	 self.Obstacle_Obj_List[i]['y'],
									  					      	 self.Obstacle_Obj_List[i]['width'],
									  					    	 self.Obstacle_Obj_List[i]['length']))

			self.Goal_candidate_num = random.randint(0,1)
			self.init = True

			self.frame_step([1, 0, 0, 0, 0])

			self.Acceleration = False
			self.Deceleration = False
			self.Left_Steering = False
			self.Right_Steering = False

			self.starttime = time.time()

	def frame_step(self, input):

		if self.init == True:
			self.Map, self.Host_Obj, self.Remote_Obj_List, self.Obstacle_Obj_List, self.score, self.Goal_candidate_num, self.starttime = self.init_condition()
			self.init = False

		score_surf = BASIC_FONT.render('Score: %.f' %self.score, True, WHITE)
		score_rect = score_surf.get_rect()
		score_rect.center = (WINDOW_WIDTH - 100, 30)

		acc_surf = BASIC_FONT.render('Acc: %.2f' %self.Host_Obj['acc'], True, WHITE)
		acc_rect = acc_surf.get_rect()
		acc_rect.center = (WINDOW_WIDTH - 100, 60)

		steering_surf = BASIC_FONT.render('Steering: %.2f' %self.Host_Obj['steering'], True, WHITE)
		steering_rect = steering_surf.get_rect()
		steering_rect.center = (WINDOW_WIDTH - 100, 90)

		DISPLAYSURF.fill(Background_Color)

		self.Draw_map(self.Map)
		self.Draw_Objs(self.Host_Obj, self.Remote_Obj_List, self.Obstacle_Obj_List)
		self.Goal_Rect = self.Draw_end_point(self.Map, self.Goal_candidate_num)

		self.Center_out = self.Is_Vehicle_Out_Center(self.Host_Obj, self.Map)
		self.Road_out = self.Is_Vehicle_Out_Road(self.Host_Obj, self.Map)
		self.Collision_check = self.Is_Collision(self.Host_Obj, self.Remote_Obj_List, self.Obstacle_Obj_List)
		self.OnLine_check = self.Is_OnLine(self.Host_Obj, self.Map)
		self.RoadEnd_check = self.Is_Roadend(self.Host_Obj, self.Map)
		self.Goal_check = self.Is_Goal(self.Host_Obj, self.Goal_Rect)

		reward = -0.001
		terminal = False

		if self.Center_out == True:
			reward = - 0.05

		if self.Road_out == True:
			reward = - 10
			self.init = True
			terminal = True
			print('Road out!')

		if self.Collision_check == True:
			reward = -10
			self.init = True
			terminal = True
			print('Collision!')

		if self.OnLine_check == True:
			reward = - 0.01

		if self.Goal_check == True:
			reward = 10
			self.init = True
			terminal = True
			print('Goal!')

		if self.RoadEnd_check == True:
			reward = -10
			self.init = True
			terminal = True 
			print('Road end!')

		# if time.time() - self.starttime > time_limit:
		# 	reward = -5
		# 	print('timeout!')
		# 	self.init = True
		# 	terminal = True

		self.score += reward

		# Event handling loop
		for event in pygame.event.get():
			if event.type == QUIT:
				terminate()

		if (input[1] == 1):
			self.Host_Obj['steering'] += 0.5

		elif (input[2] == 1):
			self.Host_Obj['steering'] -= 0.5

		elif (input[3] == 1):
			self.Host_Obj['acc'] += 0.25

		elif (input[4] == 1):
			self.Host_Obj['acc'] -= 0.25

		if self.Host_Obj['steering'] >= 20:
			self.Host_Obj['steering'] = 20

		if self.Host_Obj['steering'] <= -20:
			self.Host_Obj['steering'] = -20

		if self.Host_Obj['velocity'] >= 60:
			self.Host_Obj['velocity'] = 60

		if self.Host_Obj['velocity'] <= -30:
			self.Host_Obj['velocity'] = -30

		if self.Host_Obj['acc'] >= 9:
			self.Host_Obj['acc'] = 9

		if self.Host_Obj['acc'] <= -9:
			self.Host_Obj['acc'] = -9

		# if Acceleration == True:
		# 	Host_Obj['acc'] += 0.25
		# elif Deceleration == True:
		# 	Host_Obj['acc'] -= 0.25
		# if self.Left_Steering == True:
		# 	self.Host_Obj['steering'] += 0.5
		# elif self.Right_Steering == True:
		# 	self.Host_Obj['steering'] -= 0.5

		self.Host_Obj['heading'] += (self.Host_Obj['velocity'] * (self.deg2rad(self.Host_Obj['steering']) / self.Host_Obj['length']))/30.0
		self.Host_Obj['velocity'] += self.Host_Obj['acc'] * (1/30.0)
		
		moving_distance = self.Host_Obj['velocity'] * (1/30.0) + 0.5 * (self.Host_Obj['acc']) * ((1/30.0) ** 2)

		moving_x = moving_distance * math.cos(self.Host_Obj['heading'])
		moving_y = moving_distance * math.sin(self.Host_Obj['heading'])

		for i in range(len(self.Remote_Obj_List)):
			self.Remote_Obj_List[i]['x'] -= (self.Remote_Obj_List[i]['velocity'] * (1/30.0)) * math.sin(self.deg2rad(self.Remote_Obj_List[i]['heading']))
			self.Remote_Obj_List[i]['y'] -= (self.Remote_Obj_List[i]['velocity'] * (1/30.0)) * math.cos(self.deg2rad(self.Remote_Obj_List[i]['heading']))

		self.Host_Obj['y'] -= moving_x 
		self.Host_Obj['x'] -= moving_y

		DISPLAYSURF.blit(acc_surf, acc_rect)
		DISPLAYSURF.blit(steering_surf, steering_rect)
		DISPLAYSURF.blit(score_surf, score_rect)

		pygame.display.update()

		self.Right_Steering = False
		self.Left_Steering = False

		image_data = pygame.surfarray.array3d(pygame.display.get_surface())
		return image_data, reward, terminal
		# self.Acceleration = False
		# self.Deceleration = False


	def deg2rad(self, deg):
		return math.radians(deg)


	def terminate():
		pygame.quit()
		sys.exit()


	def init_condition(self):
		self.Map = 'Straight_UD'

		self.Host_Obj = {'surface': UP,
				 		 'x': HALF_WINDOW_WIDTH + LANE_WIDTH / 2.0,
				 		 'y': WINDOW_HEIGHT - 50.0,
						 'velocity': 10.0,
						 'acc': 0.0,
						 'steering': 0.0,
						 'heading': 0.0,
						 'width': 35,
						 'length': 80,
						 'image': HOST_IMAGE,
						 'rect': 0}

		self.Host_Obj['rect'] = pygame.Rect( (self.Host_Obj['x'],
				     				     	  self.Host_Obj['y'],
				      			         	  self.Host_Obj['width'],
				      			         	  self.Host_Obj['length']))

		self.Remote_Obj_List = []

		self.score = 0

		for i in range(RANDOM_VEHICLE_NUM):
			Init_direction_list = ['up', 'down']
			Heading_list = [0, 180]
			self.Remote_Obj_List.append({'surface': random.choice(Init_direction_list),
										 'x': random.randint(HALF_WINDOW_WIDTH - (ROAD_WIDTH / 4), HALF_WINDOW_WIDTH + (ROAD_WIDTH / 2) - 50),
								  	 	 'y': random.randint(100, WINDOW_HEIGHT - 300),
										 'velocity': random.randint(10, 40),
										 'heading': 0,
										 'image': REMOTE_IMAGES_DICT[random.choice(REMOTE_IMAGES_DICT.keys())],
										 'width': 35,
										 'length': 80,
										 'rect': 0
										 })

			self.Remote_Obj_List[i]['rect'] = pygame.Rect((self.Remote_Obj_List[i]['x'], 
													  	   self.Remote_Obj_List[i]['y'], 
													  	   self.Remote_Obj_List[i]['width'], 
													  	   self.Remote_Obj_List[i]['length']
														 ))

			if self.Remote_Obj_List[i]['surface'] == 'up':
				self.Remote_Obj_List[i]['heading'] = 0
			elif self.Remote_Obj_List[i]['surface'] == 'down':
				self.Remote_Obj_List[i]['heading'] = 180

			if self.Remote_Obj_List[i]['surface'] == 'down':
				self.Remote_Obj_List[i]['image'] = pygame.transform.rotate(self.Remote_Obj_List[i]['image'],self.Remote_Obj_List[i]['heading'])


		self.Obstacle_Obj_List = []

		for i in range(RANDOM_OBSTACLE_NUM):
			self.Obstacle_Obj_List.append({'x': random.randint(HALF_WINDOW_WIDTH, HALF_WINDOW_WIDTH + (ROAD_WIDTH / 2) - 50),
									  	   'y': random.randint(0 + 250, WINDOW_HEIGHT - 250),
									  	   'image': OBSTACLE_IMAGE,
									  	   'width': 30,
									  	   'length': 20,
									  	   'rect': 0})
			self.Obstacle_Obj_List[i]['rect'] = pygame.Rect((self.Obstacle_Obj_List[i]['x'],
								  					    	 self.Obstacle_Obj_List[i]['y'],
								  					    	 self.Obstacle_Obj_List[i]['width'],
								  					    	 self.Obstacle_Obj_List[i]['length']))

		self.Goal_candidate_num = random.randint(0,1)

		self.starttime = time.time()

		return self.Map, self.Host_Obj, self.Remote_Obj_List, self.Obstacle_Obj_List, self.score, self.Goal_candidate_num, self.starttime


	def Draw_Objs(self, Host_Obj, Remote_Obj_List, Obstacle_Obj_List):

		angle = Host_Obj['heading'] * 180 / (math.pi)
		Host_Obj_Surf = pygame.transform.rotate(Host_Obj['image'], angle)
		Host_Obj['rect'] = Host_Obj_Surf.get_rect()
		Host_Obj['rect'].center = (Host_Obj['x'], Host_Obj['y'])

		DISPLAYSURF.blit(Host_Obj_Surf, Host_Obj['rect'])

		for i in range(RANDOM_VEHICLE_NUM):
			Remote_Obj_List[i]['rect'] = pygame.Rect((Remote_Obj_List[i]['x'], 
											  		  Remote_Obj_List[i]['y'], 
											  		  Remote_Obj_List[i]['width'], 
											  		  Remote_Obj_List[i]['length']
													))
			DISPLAYSURF.blit(Remote_Obj_List[i]['image'], Remote_Obj_List[i]['rect'])

		for i in range(RANDOM_OBSTACLE_NUM):
			DISPLAYSURF.blit(Obstacle_Obj_List[i]['image'], Obstacle_Obj_List[i]['rect'])
		
		pygame.display.update()


	def Draw_map(self, map_data):
		if map_data == 'Straight_UD':
			Road_rect = pygame.draw.rect( DISPLAYSURF, Road_Color, 
				                          (HALF_WINDOW_WIDTH - ROAD_WIDTH/2,
				                           0,
				                           ROAD_WIDTH,
				                           WINDOW_HEIGHT))
			
			Center_rect = pygame.draw.rect( DISPLAYSURF, Center_Color,
										  (HALF_WINDOW_WIDTH - (LINE_WIDTH/2),
										   0,
										   LINE_WIDTH,
										   WINDOW_HEIGHT))

			Lane_rect_1 = pygame.draw.rect( DISPLAYSURF, Lane_Color,
										  (HALF_WINDOW_WIDTH - LANE_WIDTH - (LINE_WIDTH/2),
										   0,
										   LINE_WIDTH,
										   WINDOW_HEIGHT))

			Lane_rect_2 = pygame.draw.rect( DISPLAYSURF, Lane_Color,
										  (HALF_WINDOW_WIDTH + LANE_WIDTH - (LINE_WIDTH/2),
										   0,
										   LINE_WIDTH,
										   WINDOW_HEIGHT))

			Road_edge_1 = pygame.draw.rect( DISPLAYSURF, Road_Edge_Color,
										  (HALF_WINDOW_WIDTH - (ROAD_WIDTH/2) - (LINE_WIDTH/2),
										   0,
										   LINE_WIDTH,
										   WINDOW_HEIGHT))

			Road_edge_2 = pygame.draw.rect( DISPLAYSURF, Road_Edge_Color,
										  (HALF_WINDOW_WIDTH + (ROAD_WIDTH/2) - (LINE_WIDTH/2),
										   0,
										   LINE_WIDTH,
										   WINDOW_HEIGHT))

	def Draw_end_point(self, Map, Goal_candidate_num):
		if Map == 'Straight_UD':
			Goal_Candidate = [(HALF_WINDOW_WIDTH + (3*LANE_WIDTH/2), 20), 
			                  (HALF_WINDOW_WIDTH + (LANE_WIDTH/2), 20)]

		Goal_coord = Goal_Candidate[Goal_candidate_num]

		Goal_Surf = pygame.transform.scale(GOAL_IMAGE, (50,30))
		Goal_Rect = GOAL_IMAGE.get_rect()
		Goal_Rect.center = (Goal_coord[0], Goal_coord[1])

		DISPLAYSURF.blit(Goal_Surf, Goal_Rect)

		return Goal_Rect


	def Is_Vehicle_Out_Center(self, Host_Obj, Map):
		if Map == 'Straight_UD':
			if Host_Obj['x'] - Host_Obj['width']/2 < HALF_WINDOW_WIDTH + (LINE_WIDTH/2):
				return True

			return False

	def Is_Vehicle_Out_Road(self, Host_Obj, Map):
		if Map == 'Straight_UD':
			if Host_Obj['x'] - Host_Obj['width']/2 < HALF_WINDOW_WIDTH - (ROAD_WIDTH/2) + (LINE_WIDTH/2) or Host_Obj['x'] + Host_Obj['width']/2 > HALF_WINDOW_WIDTH + (ROAD_WIDTH/2) - (LINE_WIDTH/2):
				return True
			
			return False

	def Is_Collision(self, Host_Obj, Remote_Obj_List, Obstacle_Obj_List):
		for i in range(len(Remote_Obj_List)):
			Host_rect = Host_Obj['rect']
			Remote_Obj_rect = Remote_Obj_List[i]['rect']
			if Host_rect.colliderect(Remote_Obj_rect):
				return True
		for i in range(len(Obstacle_Obj_List)):
			Host_rect = Host_Obj['rect']
			Obstacle_Obj_rect = Obstacle_Obj_List[i]['rect']				
			if Host_rect.colliderect(Obstacle_Obj_rect):
				return True

		return False

	def Is_OnLine(self, Host_Obj, Map):
		if Map == 'Straight_UD':
			Lane_rect_1 = pygame.Rect( (HALF_WINDOW_WIDTH - LANE_WIDTH - (LINE_WIDTH/2),
										0,
										LINE_WIDTH,
										WINDOW_HEIGHT) )

			Lane_rect_2 = pygame.Rect( (HALF_WINDOW_WIDTH + LANE_WIDTH - (LINE_WIDTH/2),
										0,
										LINE_WIDTH,
										WINDOW_HEIGHT))

			Host_rect = Host_Obj['rect']
			# if Host_rect.colliderect(Lane_rect_1) or Host_rect.colliderect(Lane_rect_2):
			if Host_rect.colliderect(Lane_rect_2):
				return True

			return False

	def Is_Goal(self, Host_Obj, Goal_Rect):
		Host_rect = Host_Obj['rect']
		if Host_rect.colliderect(Goal_Rect):
			return True

		return False

	def Is_Roadend(self, Host_Obj, Map):
		if Map == 'Straight_UD':
			if Host_Obj['y'] <= 0:
				return True
			elif Host_Obj['y'] >= WINDOW_HEIGHT:
				return True				
		return False



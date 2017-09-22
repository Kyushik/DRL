# Dot game
# By KyushikMin kyushikmin@gmail.com
# http://mmc.hanyang.ac.kr

import random, pygame, time, sys, copy
from pygame.locals import *

FPS = 30
WINDOW_WIDTH = 200
WINDOW_HEIGHT = 240
GAME_BOARD_GAP = 20 
############## Automatic setting (later work)
GAME_BOARD_SIZE = 40
GAME_BOARD_HORIZONTAL = int((WINDOW_WIDTH - 2*GAME_BOARD_GAP) / GAME_BOARD_SIZE)
# 50 is for message  
GAME_BOARD_VERTICAL = int((WINDOW_HEIGHT - 2*GAME_BOARD_GAP - 30) / GAME_BOARD_SIZE)

# Color setting
#			      R    G    B 
WHITE 		  = (255, 255, 255)
BLACK 		  = (  0,   0,   0)
BRIGHT_RED    = (255,   0,   0)
RED 		  = (155,   0,   0)
BRIGHT_GREEN  = (  0, 255,   0)
GREEN         = (  0, 155,   0)
BRIGHT_BLUE   = (  0,   0, 255)
BLUE  		  = (  0,   0, 155)
BRIGHT_YELLOW = (255, 255,   0)
YELLOW 		  = (155, 155,   0)
DARK_GRAY 	  = ( 40,  40,  40)
LIGHT_GRAY 	  = ( 80,  80,  80)

bgColor = BLACK
gameboard_Color = BLACK
obstacle_Color = LIGHT_GRAY
text_Color = WHITE
tile_Color = LIGHT_GRAY
clicked_tile_Color = RED 
line_Color = WHITE
food_Color = GREEN
enemy_Color = RED 
my_Color = BRIGHT_BLUE

def ReturnName():
	return 'grid'

def Return_Num_Action():
    return 4
	
class GameState:
	def __init__(self):
		global FPS_CLOCK, DISPLAYSURF, BASIC_FONT

		# Set the initial variables
		pygame.init()
		FPS_CLOCK = pygame.time.Clock()
		DISPLAYSURF = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
		pygame.display.set_caption('Dot')
		BASIC_FONT = pygame.font.Font('freesansbold.ttf', 18)
		
		Movement_list = ['North', 'South', 'West', 'East', 'Stop']
		difficulty = 'Easy'

		#set up the variables
		self.score = 0
		self.Game_board_state = []
		self.Coordinate_info = []
		self.My_position = []
		self.Enemy_list = []
		self.Food_list = []
		self.Last_enemy_move = []
		
		self.Game_board_state, self.Coordinate_info = self.drawGameBoard(difficulty)
		self.checkForQuit()
		self.drawBasicBoard()		

		self.count_init = 0
		self.reward_food = 1
		self.reward_enemy = -1

		self.count_food = 0

		self.result = ''

	def reinit(self):
		#set up the variables
		self.Last_enemy_move = []

		difficulty = 'Easy'

		self.Game_board_state, self.Coordinate_info = self.drawGameBoard(difficulty)
		self.checkForQuit()
		
		self.drawBasicBoard()		
		self.count_init = 0
		self.count_food = 0
		
	# Main function
	def frame_step(self, input):									
		self.checkForQuit()
		###################### Game display ######################
		DISPLAYSURF.fill(bgColor)
		terminal = False

		# scoreSurf = BASIC_FONT.render('Result: ' + str(self.score), 1, WHITE)
		# scoreRect = scoreSurf.get_rect()
		# scoreRect.topleft = (WINDOW_WIDTH - 200, 10)
		
		self.checkForQuit()

		if self.count_init == 0:
			# Initialize my position
			self.My_position = self.Get_random_position()
			self.Game_board_state[self.My_position[1]][self.My_position[0]] = '@' 

			# Initialize enemy position
			self.Enemy_list = [self.Get_random_position()]
			for i in range(len(self.Enemy_list)):
				self.Game_board_state[self.Enemy_list[i][1]][self.Enemy_list[i][0]] = '-'
			
			# Initialize food position
			self.Food_list = [self.Get_random_position()]
			for i in range(len(self.Food_list)):
				self.Game_board_state[self.Food_list[i][1]][self.Food_list[i][0]] = '+'			
		
		self.DrawGameBoardState()
		self.Drawlines()

		# if (input[1] == 1) and  ('North' in self.ValidMove_list((self.My_position[0], self.My_position[1]))):
		# 	self.Game_board_state[self.My_position[1] - 1][self.My_position[0]] = '@'
		# 	self.Game_board_state[self.My_position[1]][self.My_position[0]] = 0
		# 	self.My_position[1] = self.My_position[1] - 1
			
		# elif (input[0] == 1) and  ('South' in self.ValidMove_list((self.My_position[0], self.My_position[1]))):
		# 	self.Game_board_state[self.My_position[1] + 1][self.My_position[0]] = '@'
		# 	self.Game_board_state[self.My_position[1]][self.My_position[0]] = 0
		# 	self.My_position[1] = self.My_position[1] + 1
			
		# elif (input[2] == 1) and  ('East' in self.ValidMove_list((self.My_position[0], self.My_position[1]))):
		# 	self.Game_board_state[self.My_position[1]][self.My_position[0] + 1] = '@'
		# 	self.Game_board_state[self.My_position[1]][self.My_position[0]] = 0
		# 	self.My_position[0] = self.My_position[0] + 1
			
		# elif (input[3] == 1) and  ('West' in self.ValidMove_list((self.My_position[0], self.My_position[1]))):
		# 	self.Game_board_state[self.My_position[1]][self.My_position[0] - 1] = '@'
		# 	self.Game_board_state[self.My_position[1]][self.My_position[0]] = 0
		# 	self.My_position[0] = self.My_position[0] - 1

		# North
		if input[1] == 1:
			self.Game_board_state[self.My_position[1]][self.My_position[0]] = 0
			if self.My_position[1] == 0:
				self.Game_board_state[3][self.My_position[0]] = '@'
				self.My_position[1] = 3
			else:
				self.Game_board_state[self.My_position[1] - 1][self.My_position[0]] = '@'
				self.My_position[1] = self.My_position[1] - 1
		
		# South
		elif input[0] == 1:
			self.Game_board_state[self.My_position[1]][self.My_position[0]] = 0
			if self.My_position[1] == 3:
				self.Game_board_state[0][self.My_position[0]] = '@'
				self.My_position[1] = 0
			else:
				self.Game_board_state[self.My_position[1] + 1][self.My_position[0]] = '@'
				self.My_position[1] = self.My_position[1] + 1
		
		# East
		elif input[2] == 1:
			self.Game_board_state[self.My_position[1]][self.My_position[0]] = 0
			if self.My_position[0] == 3:
				self.Game_board_state[self.My_position[1]][0] = '@'
				self.My_position[0] = 0
			else:
				self.Game_board_state[self.My_position[1]][self.My_position[0] + 1] = '@'
				self.My_position[0] = self.My_position[0] + 1
		
		# West
		elif input[3] == 1:
			self.Game_board_state[self.My_position[1]][self.My_position[0]] = 0

			if self.My_position[0] == 0:
				self.Game_board_state[self.My_position[1]][3] = '@'	
				self.My_position[0] = 3
			else:
				self.Game_board_state[self.My_position[1]][self.My_position[0] - 1] = '@'
				self.My_position[0] = self.My_position[0] - 1

		reward = -0.01

		#Draw enemy 
		for i in range(len(self.Enemy_list)):
			self.Game_board_state[self.Enemy_list[i][1]][self.Enemy_list[i][0]] = '-'

		self.checkForQuit()

		# Draw food
		for i in range(len(self.Food_list)):
			self.Game_board_state[self.Food_list[i][1]][self.Food_list[i][0]] = '+'

		# Eat the foods 
		if self.My_position in self.Food_list:
			reward = self.reward_food
			self.score += 1.0
			self.count_food += 1

			state = pygame.surfarray.array3d(pygame.display.get_surface())
			terminal = True
			self.result = 'Win'

			self.reinit()
			return state, reward, terminal

		# Killed by enemy
		if self.My_position in self.Enemy_list:
			reward = self.reward_enemy
			self.score -= 1
			state = pygame.surfarray.array3d(pygame.display.get_surface())
			terminal = True 
			self.result = 'Lose'

			self.reinit()
			return state, reward, terminal

		score_SURF, score_RECT = self.makeText('Result: ' + str(self.result) + '      ', WHITE, BLACK, WINDOW_WIDTH - 180, 15)
		DISPLAYSURF.blit(score_SURF, score_RECT)

		pygame.display.update()
		self.checkForQuit()

		self.count_init += 1

		state = pygame.surfarray.array3d(pygame.display.get_surface())
		return state, reward, terminal

	def terminate(self):
		pygame.quit()
		sys.exit()

	def checkForQuit(self):
		for event in pygame.event.get(QUIT): # Bring every QUIT event
			terminate() # Exit then event occured
		for event in pygame.event.get(KEYUP): # Bring every KEYUP event
			if event.key == K_ESCAPE:
				terminate() # if KEYUP event is ESC then quit
			pygame.event.post(event) # Other KEYUP event object is returned to event que

	def makeText(self,text, color, bgcolor, top, left):
		# Show surface object, Rect object
		textSurf = BASIC_FONT.render(text, True, color, bgcolor)
		textRect = textSurf.get_rect()
		textRect.topleft = (top, left)
		return (textSurf, textRect)

	def drawBasicBoard(self):
		for i in range(GAME_BOARD_HORIZONTAL+1):		
			for j in range(GAME_BOARD_VERTICAL+1):
				pygame.draw.rect(DISPLAYSURF, gameboard_Color, (GAME_BOARD_GAP + i * GAME_BOARD_SIZE, 50 + GAME_BOARD_GAP + j * GAME_BOARD_SIZE, GAME_BOARD_SIZE, GAME_BOARD_SIZE))

	
	def Drawlines(self):
		for i in range(GAME_BOARD_HORIZONTAL+1):		
			for j in range(GAME_BOARD_VERTICAL+1):
				pygame.draw.line(DISPLAYSURF, line_Color, (GAME_BOARD_GAP + i * GAME_BOARD_SIZE, GAME_BOARD_GAP + 50),(GAME_BOARD_GAP + i * GAME_BOARD_SIZE, 50 + GAME_BOARD_GAP + GAME_BOARD_VERTICAL * GAME_BOARD_SIZE),2)
				pygame.draw.line(DISPLAYSURF, line_Color, (GAME_BOARD_GAP, 50 + GAME_BOARD_GAP + j * GAME_BOARD_SIZE), (GAME_BOARD_GAP + GAME_BOARD_HORIZONTAL * GAME_BOARD_SIZE, 50 + GAME_BOARD_GAP + j * GAME_BOARD_SIZE),2)
							
	def drawGameBoard(self,difficulty):
		if difficulty == 'Easy':
			Game_board_state = [[ 0, 0, 0, 0 ],\
								[ 0, 0, 0, 0 ],\
								[ 0, 0, 0, 0 ],\
								[ 0, 0, 0, 0 ]]

		# Add coordinate info
		Coordinate_info = [[],[],[]]
		for i in range(GAME_BOARD_HORIZONTAL):		
			for j in range(GAME_BOARD_VERTICAL):

				center_point = (GAME_BOARD_GAP + i * GAME_BOARD_SIZE + GAME_BOARD_SIZE/2 + 1, 50 + GAME_BOARD_GAP + j * GAME_BOARD_SIZE + GAME_BOARD_SIZE/2 + 1)
				radius = GAME_BOARD_SIZE/2  - 2

				if Game_board_state[j][i] == 1:
					pygame.draw.rect(DISPLAYSURF, obstacle_Color, (GAME_BOARD_GAP + i * GAME_BOARD_SIZE, 50 + GAME_BOARD_GAP + j * GAME_BOARD_SIZE, GAME_BOARD_SIZE, GAME_BOARD_SIZE))
				elif Game_board_state[j][i] == '+':
					pygame.draw.polygon(DISPLAYSURF, food_Color, ((center_point[0], center_point[1] + radius), (center_point[0] + radius, center_point[1]), (center_point[0], center_point[1] - radius), (center_point[0] - radius, center_point[1])), 10)
					Coordinate_info[2].append([i,j])
				elif Game_board_state[j][i] == '-':
					Coordinate_info[1].append([i,j])
					pygame.draw.rect(DISPLAYSURF, enemy_Color, (GAME_BOARD_GAP + i * GAME_BOARD_SIZE + 5, 50 + GAME_BOARD_GAP + j * GAME_BOARD_SIZE + 5, GAME_BOARD_SIZE - 5, GAME_BOARD_SIZE - 5))
				elif Game_board_state[j][i] == '@':
					pygame.draw.rect(DISPLAYSURF, my_Color, (GAME_BOARD_GAP + i * GAME_BOARD_SIZE + 5, 50 + GAME_BOARD_GAP + j * GAME_BOARD_SIZE + 5, GAME_BOARD_SIZE - 5, GAME_BOARD_SIZE - 5))
					Coordinate_info[0].append([i,j])
				
		pygame.display.update()			
		return Game_board_state, Coordinate_info

	def DrawGameBoardState(self):
		for i in range(GAME_BOARD_HORIZONTAL):		
			for j in range(GAME_BOARD_VERTICAL):

				center_point = (GAME_BOARD_GAP + i * GAME_BOARD_SIZE + GAME_BOARD_SIZE/2 + 1, 50 + GAME_BOARD_GAP + j * GAME_BOARD_SIZE + GAME_BOARD_SIZE/2 + 1)
				radius = GAME_BOARD_SIZE/2  - 2

				if self.Game_board_state[j][i] == 1:
					pygame.draw.rect(DISPLAYSURF, obstacle_Color, (GAME_BOARD_GAP + i * GAME_BOARD_SIZE, 50 + GAME_BOARD_GAP + j * GAME_BOARD_SIZE, GAME_BOARD_SIZE, GAME_BOARD_SIZE))
				elif self.Game_board_state[j][i] == '+':
					pygame.draw.polygon(DISPLAYSURF, food_Color, ((center_point[0], center_point[1] + radius - 3), (center_point[0] + radius - 3, center_point[1]), (center_point[0], center_point[1] - radius + 3), (center_point[0] - radius + 3, center_point[1])), 10)
				elif self.Game_board_state[j][i] == '-':
					pygame.draw.rect(DISPLAYSURF, enemy_Color, (GAME_BOARD_GAP + i * GAME_BOARD_SIZE + 5, 50 + GAME_BOARD_GAP + j * GAME_BOARD_SIZE + 5, GAME_BOARD_SIZE - 10, GAME_BOARD_SIZE - 10))
				elif self.Game_board_state[j][i] == '@':
					pygame.draw.rect(DISPLAYSURF, my_Color, (GAME_BOARD_GAP + i * GAME_BOARD_SIZE + 5, 50 + GAME_BOARD_GAP + j * GAME_BOARD_SIZE + 5, GAME_BOARD_SIZE - 5, GAME_BOARD_SIZE - 5))
				
		pygame.display.update()			

	def ValidMove_list(self, state):
    	# return the valid move( no obstacles and no out of bound)
		state_x = state[0]
		state_y = state[1]
		valid_move = []
		if state_y + 1 <= GAME_BOARD_VERTICAL - 1 and self.Game_board_state[state_y + 1][state_x] != 1:
			valid_move.append('South')
		if state_y -1 >= 0 and self.Game_board_state[state_y - 1][state_x] != 1:
			valid_move.append('North')
		if state_x - 1 >= 0 and self.Game_board_state[state_y][state_x - 1] != 1:
			valid_move.append('West')
		if state_x + 1 <= GAME_BOARD_HORIZONTAL - 1 and self.Game_board_state[state_y][state_x + 1] != 1:
			valid_move.append('East')
		valid_move.append('Stop')

		return valid_move

	def Get_random_position(self):
		while True:
			random_x = random.randint(0,GAME_BOARD_HORIZONTAL-1)
			random_y = random.randint(0,GAME_BOARD_VERTICAL-1)

			if self.Game_board_state[random_y][random_x] == 0:
				return [random_x, random_y]
				break

if __name__ == '__main__':
	main()
# Atari breakout
# By KyushikMin kyushikmin@gamil.com
# http://mmc.hanyang.ac.kr
# Special thanks to my colleague Hayoung and Jongwon for giving me the idea of ball and block collision algorithm

import random, sys, time, math, pygame
from pygame.locals import *
import numpy as np
import copy

# Window Information
FPS = 30
WINDOW_WIDTH = 480
WINDOW_HEIGHT = 400

INFO_GAP  = 40
UPPER_GAP = 40
HALF_WINDOW_WIDTH = int(WINDOW_WIDTH / 2)
HALF_WINDOW_HEIGHT = int((WINDOW_HEIGHT - INFO_GAP) / 2)

# Colors
#				 R    G    B
WHITE        = (255, 255, 255)
BLACK		 = (  0,   0,   0)
RED 		 = (200,  72,  72)
LIGHT_ORANGE = (198, 108,  58)
ORANGE       = (180, 122,  48)
GREEN		 = ( 72, 160,  72)
BLUE 		 = ( 66,  72, 200)
YELLOW 		 = (162, 162,  42)
NAVY         = ( 75,   0, 130)
PURPLE       = (143,   0, 255)

bar_width = 60
bar_height = 8
bar_speed1 = 5
bar_speed2 = 10
bar_init_position = (WINDOW_WIDTH - bar_width)/2

ball_init_position_x = WINDOW_WIDTH / 2
ball_init_position_y = (WINDOW_HEIGHT - INFO_GAP) / 2 + UPPER_GAP
ball_radius = 5
ball_bounce_speed_range = 10

block_width  = 48
block_height = 18

num_block_row = int(((WINDOW_HEIGHT - INFO_GAP) / 4) / block_height) # Number of rows should be less than 8 or you should add more colors
num_block_col = int(WINDOW_WIDTH / block_width)

block_color_list = [RED, LIGHT_ORANGE, YELLOW, GREEN, BLUE, NAVY, PURPLE]

def ReturnName():
	return 'breakout'

def Return_Num_Action():
    return 5

class GameState:
	def __init__(self):
		global FPS_CLOCK, DISPLAYSURF, BASIC_FONT

		# Set the initial variables
		pygame.init()
		FPS_CLOCK = pygame.time.Clock()

		DISPLAYSURF = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

		pygame.display.set_caption('BreakOut')
		# pygame.display.set_icon(pygame.image.load('./Qar_Sim/icon_resize2.png'))

		BASIC_FONT = pygame.font.Font('freesansbold.ttf', 16)

		# Set initial parameters
		self.init = True
		self.score = 0
		self.reward = 0

		self.num_blocks = num_block_row * num_block_col

		self.init_block_info = []
		for i in range(num_block_row):
			self.init_block_info.append([])
			for j in range(num_block_col):
				self.init_block_info[i].append([])

		for i in range(num_block_row):
			for j in range(num_block_col):
				# Horizontal position, Vertical position, Width, Height
				self.init_block_info[i][j] = [(j * block_width, UPPER_GAP + INFO_GAP + i * block_height, block_width, block_height), 'visible']

		self.direction = ''

	# Main function
	def frame_step(self, input):
		# Initial settings
		reward = 0
		terminal = False

		if self.init == True:
			self.bar_position = bar_init_position
			self.ball_position_x = ball_init_position_x
			self.ball_position_y = ball_init_position_y

			# self.ball_speed_x = random.randint(-3, 3)
			self.ball_speed_x = random.uniform(-3.0, 3.0)
			self.ball_speed_y = 5

			self.block_info = copy.deepcopy(self.init_block_info)

			self.init = False

		# Key settings
		for event in pygame.event.get(): # event loop
			if event.type == QUIT:
				self.terminate()

		if input[1] == 1:
			self.bar_position += bar_speed1 # slow right
		elif input[2] == 1:
			self.bar_position += bar_speed2 # fast right
		elif input[3] == 1:
			self.bar_position -= bar_speed1 # slow left
		elif input[4] == 1:
			self.bar_position -= bar_speed2 # fast left


		# Constraint of the bar
		if self.bar_position <= 0:
			self.bar_position = 0

		if self.bar_position >= WINDOW_WIDTH - bar_width:
			self.bar_position = WINDOW_WIDTH - bar_width

		# Move the ball
		self.ball_position_x += self.ball_speed_x
		self.ball_position_y += self.ball_speed_y

		# Ball is bounced when the ball hit the wall
		if self.ball_position_x < ball_radius:
			self.ball_speed_x = - self.ball_speed_x
			self.ball_position_x = ball_radius

		if self.ball_position_x >= WINDOW_WIDTH - ball_radius:
			self.ball_speed_x = - self.ball_speed_x
			self.ball_position_x = WINDOW_WIDTH - ball_radius

		if self.ball_position_y < INFO_GAP + ball_radius:
			self.ball_speed_y = - self.ball_speed_y
			self.ball_position_y = INFO_GAP + ball_radius

		# Ball is bounced when the ball hit the bar
		if self.ball_position_y >= WINDOW_HEIGHT - bar_height - ball_radius:
			# Hit the ball!
			if self.ball_position_x <= self.bar_position + bar_width and self.ball_position_x >= self.bar_position:
				ball_hit_point = self.ball_position_x - self.bar_position
				ball_hit_point_ratio = ball_hit_point / bar_width
				self.ball_speed_x = (ball_hit_point_ratio * ball_bounce_speed_range) - (ball_bounce_speed_range/2)

				if abs(ball_hit_point_ratio - 0.5) < 0.01:
					self.ball_speed_x = random.uniform(-0.01 * ball_bounce_speed_range/2 , 0.01 * ball_bounce_speed_range/2)

				self.ball_speed_y = - self.ball_speed_y
				self.ball_position_y = WINDOW_HEIGHT - bar_height - ball_radius
				# reward = 0.5

		# Lose :(
		if self.ball_position_y >= WINDOW_HEIGHT:
			self.init = True
			reward = -1
			terminal = True

		# When the ball hit the block
		check_ball_hit_block = 0
		for i in range(num_block_row):
			for j in range(num_block_col):
				block_left  = self.block_info[i][j][0][0]
				block_right = self.block_info[i][j][0][0] + self.block_info[i][j][0][2]
				block_up    = self.block_info[i][j][0][1]
				block_down  = self.block_info[i][j][0][1] + self.block_info[i][j][0][3]
				visible     = self.block_info[i][j][1]

				# The ball hit some block!!
				# if (block_left <= self.ball_position_x + ball_radius) and (self.ball_position_x - ball_radius <= block_right) and (block_up <= self.ball_position_y + ball_radius) and (self.ball_position_y - ball_radius <= block_down) and visible == 'visible':
				if (block_left <= self.ball_position_x) and (self.ball_position_x <= block_right) and (block_up <= self.ball_position_y) and (self.ball_position_y <= block_down) and visible == 'visible':
					# Which part of the block was hit??
					# Upper left, Upper right, Lower right, Lower left
					block_points = [[block_left, block_up], [block_right, block_up], [block_right, block_down], [block_left, block_down]]

					if self.ball_position_x -self. ball_position_x_old == 0:
						slope_ball = (self.ball_position_y - self.ball_position_y_old) / (0.1)
					else:
						slope_ball = (self.ball_position_y - self.ball_position_y_old) / (self.ball_position_x - self.ball_position_x_old)

					# ax+by+c = 0
					line_coeff = [slope_ball, -1, self.ball_position_y_old - (slope_ball * self.ball_position_x_old)]

					point1 = [block_left, (-1/line_coeff[1]) * (line_coeff[0] * block_left + line_coeff[2])]
					point2 = [block_right, (-1/line_coeff[1]) * (line_coeff[0] * block_right + line_coeff[2])]
					point3 = [(-1/line_coeff[0]) * (line_coeff[1] * block_up + line_coeff[2]), block_up]
					point4 = [(-1/line_coeff[0]) * (line_coeff[1] * block_down + line_coeff[2]), block_down]

					# Left, Right, Up, Down
					intersection = [point1, point2, point3, point4]
					check_intersection = [0, 0, 0, 0]

					for k in range(len(intersection)):
						#intersection point is on the left side of block
						if intersection[k][0] == block_left and (block_up <= intersection[k][1] <= block_down):
							check_intersection[0] = 1

						if intersection[k][0] == block_right and (block_up <= intersection[k][1] <= block_down):
							check_intersection[1] = 1

						if intersection[k][1] == block_up and (block_left <= intersection[k][0] <= block_right):
							check_intersection[2] = 1

						if intersection[k][1] == block_down and (block_left <= intersection[k][0] <= block_right):
							check_intersection[3] = 1

					dist_points = [np.inf, np.inf, np.inf, np.inf]
					for k in range(len(intersection)):
						if check_intersection[k] == 1:
							dist = self.get_dist(intersection[k], [self.ball_position_x_old, self.ball_position_y_old])
							dist_points[k] = dist

					# 0: Left, 1: Right, 2: Up, 3: Down
					collision_line = np.argmin(dist_points)

					if collision_line == 0:
						self.ball_speed_x = - self.ball_speed_x
					elif collision_line == 1:
						self.ball_speed_x = - self.ball_speed_x
					elif collision_line == 2:
						self.ball_speed_y = - self.ball_speed_y
					elif collision_line == 3:
						self.ball_speed_y = - self.ball_speed_y

					# Incorrect breaking at corner!
					# e.g. block was hit on the right side even though there is visible block on the right
					# Then, the former decision was wrong, so change the direction!
					if j > 0:
						if collision_line == 0 and self.block_info[i][j-1][1] == 'visible':
							self.ball_speed_x = - self.ball_speed_x
							self.ball_speed_y = - self.ball_speed_y
					if j < num_block_col - 1:
						if collision_line == 1 and self.block_info[i][j+1][1] == 'visible':
							self.ball_speed_x = - self.ball_speed_x
							self.ball_speed_y = - self.ball_speed_y
					if i > 0:
						if collision_line == 2 and self.block_info[i-1][j][1] == 'visible':
							self.ball_speed_x = - self.ball_speed_x
							self.ball_speed_y = - self.ball_speed_y
					if i < num_block_row - 1:
						if collision_line == 3 and self.block_info[i+1][j][1] == 'visible':
							self.ball_speed_x = - self.ball_speed_x
							self.ball_speed_y = - self.ball_speed_y

					# Move the ball to the block boundary after ball hit the block
					if collision_line == 0:
						self.ball_position_x = block_left - ball_radius
					elif collision_line == 1:
						self.ball_position_x = block_right + ball_radius
					elif collision_line == 2:
						self.ball_position_y = block_up - ball_radius
					elif collision_line == 3:
						self.ball_position_y = block_down + ball_radius

					# make hit block invisible
					self.block_info[i][j][1] = 'invisible'
					check_ball_hit_block = 1
					reward = 1

				# If one block is hitted, break the for loop (Preventing to break multiple blocks at once)
				if check_ball_hit_block == 1:
					break
			# If one block is hitted, break the for loop (Preventing to break multiple blocks at once)
			if check_ball_hit_block == 1:
				break

		# Fill background color
		DISPLAYSURF.fill(BLACK)

		# Draw blocks
		count_visible = 0
		for i in range(num_block_row):
			for j in range(num_block_col):
				if self.block_info[i][j][1] == 'visible':
					pygame.draw.rect(DISPLAYSURF, block_color_list[i], self.block_info[i][j][0])
					count_visible += 1

		# Win the game!! :)
		if count_visible == 0:
			self.init = True
			reward = 11
			terminal = True

		# Display informations
		score_value = self.num_blocks - count_visible
		self.score_msg(score_value)
		self.block_num_msg(count_visible)

		# Draw bar
		bar_rect = pygame.Rect(self.bar_position, WINDOW_HEIGHT - bar_height, bar_width, bar_height)
		pygame.draw.rect(DISPLAYSURF, RED, bar_rect)

		self.ball_position_x_old = self.ball_position_x
		self.ball_position_y_old = self.ball_position_y

		# Draw ball
		pygame.draw.circle(DISPLAYSURF, WHITE, (int(self.ball_position_x), int(self.ball_position_y)), ball_radius, 0)

		# Draw line for seperate game and info
		pygame.draw.line(DISPLAYSURF, WHITE, (0, 40), (WINDOW_WIDTH, 40), 3)

		pygame.display.update()
		image_data = pygame.surfarray.array3d(pygame.display.get_surface())
		return image_data, reward, terminal

	# Exit the game
	def terminate(self):
		pygame.quit()
		sys.exit()

	# Display score
	def score_msg(self, score):
		scoreSurf = BASIC_FONT.render('Score: ' + str(score), True, WHITE)
		scoreRect = scoreSurf.get_rect()
		scoreRect.topleft = (10, 10)
		DISPLAYSURF.blit(scoreSurf, scoreRect)

	# Display how many blocks are left
	def block_num_msg(self, num_blocks):
		blockNumSurf = BASIC_FONT.render('Number of Blocks: ' + str(num_blocks), True, WHITE)
		blockNumRect = blockNumSurf.get_rect()
		blockNumRect.topleft = (WINDOW_WIDTH - 180, 10)
		DISPLAYSURF.blit(blockNumSurf, blockNumRect)

	def get_dist(self, point1, point2):
		return math.sqrt(math.pow(point1[0] - point2[0],2) + math.pow(point1[1] - point2[1], 2))

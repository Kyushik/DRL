# Modified from Tetromino by Al Sweigart al@inventwithpython.com
# http://inventwithpython.com/pygame
# Released under a "Simplified BSD" license

import random, pygame, sys
from pygame.locals import *

FPS = 30
WINDOW_WIDTH = 320
WINDOW_HEIGHT = 240
CELL_SIZE = 20
assert WINDOW_WIDTH % CELL_SIZE == 0, "Window width must be a multiple of cell size"
assert WINDOW_HEIGHT % CELL_SIZE == 0, "Window height must be a multiple of cell size"
CELL_WIDTH = int(WINDOW_WIDTH / CELL_SIZE)
CELL_HEIGHT = int(WINDOW_HEIGHT / CELL_SIZE)

#			  R    G    B 
WHITE 	  = (255, 255, 255)
BLACK 	  = (  0,   0,   0)
RED       = (255,   0,   0)
GREEN     = (  0, 255,   0)
DARKGREEN = (  0, 155,   0)
DARKGRAY  = ( 40,  40,  40)
BG_COLOR = BLACK

def ReturnName():
    return 'wormy'

def Return_Num_Action():
    return 4

class GameState:
	def __init__(self):
		global FPS_CLOCK, DISPLAYSURF, BASIC_FONT

		pygame.init()
		FPS_CLOCK = pygame.time.Clock()
		DISPLAYSURF = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
		BASIC_FONT = pygame.font.Font('freesansbold.ttf', 18)
		pygame.display.set_caption('Wormy')

		# Set the random start position
		self.startx = random.randint(5, CELL_WIDTH - 6)
		self.starty = random.randint(5, CELL_HEIGHT - 6)

		self.wormCoords = [{'x': self.startx,     'y': self.starty},
						   {'x': self.startx - 1, 'y': self.starty},
						   {'x': self.startx - 2, 'y': self.starty}]

		self.UP = 'up'
		self.DOWN = 'down'
		self.LEFT = 'left'
		self.RIGHT = 'right'

		self.direction = self.RIGHT

		# Set the apple at the random position
		self.apple = self.getRandomLocation()

		self.HEAD = 0 # Synthetic sugar: index of worm's head

	def reinit(self):
		# Set the random start position
		self.startx = random.randint(5, CELL_WIDTH - 6)
		self.starty = random.randint(5, CELL_HEIGHT - 6)

		self.wormCoords = [{'x': self.startx,     'y': self.starty},
						   {'x': self.startx - 1, 'y': self.starty},
						   {'x': self.startx - 2, 'y': self.starty}]
		self.direction = self.RIGHT

		# Set the apple at the random position
		self.apple = self.getRandomLocation()

		self.HEAD = 0 # Synthetic sugar: index of worm's head

	def frame_step(self, input):
		
		# Choose action according to the input vector
		if input[0] == 1 and self.direction != self.DOWN:
			self.direction = self.UP
		elif input[1] == 1 and self.direction != self.UP:
			self.direction = self.DOWN 
		elif input[2] == 1 and self.direction != self.RIGHT:
			self.direction = self.LEFT 
		elif input[3] == 1 and self.direction != self.LEFT:
			self.direction = self.RIGHT

		for event in pygame.event.get(): # event loop
			if event.type == QUIT:
				self.terminate()
		
		# If there is no event, reward is -0.01 and terminal is False
		reward = -0.01 
		terminal = False
		# Wormy is Dead!! minus reward and terminal is True!!
		# Check that worm crashes itself or wall
		if self.wormCoords[self.HEAD]['x'] == -1 or self.wormCoords[self.HEAD]['x'] == CELL_WIDTH or self.wormCoords[self.HEAD]['y'] == -1 or self.wormCoords[self.HEAD]['y'] == CELL_HEIGHT: # Crash with wall
			reward = -10
			terminal = True

			image_data = pygame.surfarray.array3d(pygame.display.get_surface())

			self.reinit()
			pygame.display.update()
			return image_data, reward, terminal

		for wormBody in self.wormCoords[1:]:
			if wormBody['x'] == self.wormCoords[self.HEAD]['x'] and wormBody['y'] == self.wormCoords[self.HEAD]['y']:
				reward = -10
				terminal = True 

				image_data = pygame.surfarray.array3d(pygame.display.get_surface())

				self.reinit()
				pygame.display.update()
				return image_data, reward, terminal

		# Check that worm eats the apple
		if self.wormCoords[self.HEAD]['x'] == self.apple['x'] and self.wormCoords[self.HEAD]['y'] == self.apple['y']:
			# Don't erase the worm's tail part
			reward = 1
			self.apple = self.getRandomLocation() # Put new apple again
		else:
			del self.wormCoords[-1] # Erase the worm's tail part

		# Move the worm with add the part on the moving direction
		if self.direction == self.UP:
			newHead = {'x': self.wormCoords[self.HEAD]['x'], 'y': self.wormCoords[self.HEAD]['y'] - 1}
		elif self.direction == self.DOWN:
			newHead = {'x': self.wormCoords[self.HEAD]['x'], 'y': self.wormCoords[self.HEAD]['y'] + 1}
		elif self.direction == self.LEFT:
			newHead = {'x': self.wormCoords[self.HEAD]['x'] - 1, 'y': self.wormCoords[self.HEAD]['y']}
		elif self.direction == self.RIGHT:
			newHead = {'x': self.wormCoords[self.HEAD]['x'] + 1, 'y': self.wormCoords[self.HEAD]['y']}
		
		self.wormCoords.insert(0, newHead)
		
		DISPLAYSURF.fill(BG_COLOR)
		
		self.drawGrid()
		self.drawWorm(self.wormCoords)
		self.drawApple(self.apple)
		self.drawScore(len(self.wormCoords) - 3)

		pygame.display.update()

		image_data = pygame.surfarray.array3d(pygame.display.get_surface())
		return image_data, reward, terminal

		# FPS_CLOCK.tick(FPS)


	def terminate(self):
		pygame.quit()
		sys.exit()


	def getRandomLocation(self):
		return {'x': random.randint(0, CELL_WIDTH - 1), 'y': random.randint(0, CELL_HEIGHT - 1)}

	def drawScore(self, score):
		scoreSurf = BASIC_FONT.render('Score: %s' % (score), True, WHITE)
		scoreRect = scoreSurf.get_rect()
		scoreRect.topleft = (WINDOW_WIDTH - 120, 10)
		DISPLAYSURF.blit(scoreSurf, scoreRect)


	def drawWorm(self, wormCoords):
		for coord in wormCoords:
			x = coord['x'] * CELL_SIZE
			y = coord['y'] * CELL_SIZE
			wormSegmentRect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
			pygame.draw.rect(DISPLAYSURF, DARKGREEN, wormSegmentRect)
			wormInnerSegmentRect = pygame.Rect(x + 4, y + 4, CELL_SIZE - 8, CELL_SIZE - 8)
			pygame.draw.rect(DISPLAYSURF, GREEN, wormInnerSegmentRect)


	def drawApple(self, coord):
		x = coord['x'] * CELL_SIZE
		y = coord['y'] * CELL_SIZE
		appleRect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
		pygame.draw.rect(DISPLAYSURF, RED, appleRect)


	def drawGrid(self):
		for x in range(0, WINDOW_WIDTH, CELL_SIZE): # Draw vertical lines
			pygame.draw.line(DISPLAYSURF, DARKGRAY, (x, 0), (x, WINDOW_HEIGHT))
		for y in range(0, WINDOW_HEIGHT, CELL_SIZE): # Draw horizontal lines
			pygame.draw.line(DISPLAYSURF, DARKGRAY, (0, y), (WINDOW_WIDTH, y))


if __name__ == '__main__':
	main()







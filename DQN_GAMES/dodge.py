# Dodge

# Rule
'''
Blue ball is agent and red ball is enemy.
If agent collide with enemy, game is over
Evade the enemy as long as possible!!
'''

# By KyushikMin kyushikmin@gamil.com
# http://mmc.hanyang.ac.kr

import random, sys, time, math, pygame
from pygame.locals import *
import numpy as np
import copy

# Window Information
FPS = 30

GAP_WIDTH = 10
TOP_WIDTH = 40

WINDOW_WIDTH = 360
WINDOW_HEIGHT = WINDOW_WIDTH + TOP_WIDTH

HALF_WINDOW_WIDTH = int(WINDOW_WIDTH / 2)
HALF_WINDOW_HEIGHT = int(WINDOW_HEIGHT / 2)

CENTER_X = int(WINDOW_WIDTH / 2)
CENTER_Y = int(TOP_WIDTH + (WINDOW_HEIGHT - TOP_WIDTH)/2)

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

def ReturnName():
    return 'dodge'

def Return_Num_Action():
    return 5

class GameState:
    def __init__(self):
        global FPS_CLOCK, DISPLAYSURF, BASIC_FONT

        pygame.init()
        FPS_CLOCK = pygame.time.Clock()

        DISPLAYSURF = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

        pygame.display.set_caption('Dodge')

        BASIC_FONT = pygame.font.Font('freesansbold.ttf', 16)

        # Set initial parameters

        self.init = True
        self.start_time = time.time()

        self.my_radius = 10
        self.my_init_position = [CENTER_X - int(self.my_radius/2), CENTER_Y - int(self.my_radius/2)]
        self.my_position = self.my_init_position
        self.my_speed = 10

        self.num_balls = 5
        self.gap_balls = 50

        # Set ball position and velocity
        # Ball_list: ID, x_position, y_position, x_velocity, y_velocity
        self.min_ball_speed = 3.0
        self.max_ball_speed = 6.0

        self.ball_list = self.set_ball_pos_and_vel()
        self.ball_radius = 5

    def frame_step(self, input): # Game loop
        # Initial settings
        if self.init == True:
            self.my_position = [CENTER_X - int(self.my_radius/2), CENTER_Y - int(self.my_radius/2)]
            self.ball_list = self.set_ball_pos_and_vel()

            self.start_time = time.time()

            self.init = False

        # Key settings
        for event in pygame.event.get(): # event loop
            if event.type == QUIT:
                terminate()

        # Control the bar
        if input[1] == 1:
            self.my_position[1] -= self.my_speed
        elif input[2] == 1:
            self.my_position[1] += self.my_speed
        elif input[3] == 1:
            self.my_position[0] -= self.my_speed
        elif input[4] == 1:
            self.my_position[0] += self.my_speed

        # Constraint of the agent
        self.constraint()

        # Update ball
        self.update_balls()

        # Lose :(
        is_lose = self.check_lose()

        if is_lose:
            terminal = True
            reward = -1
        else:
            terminal = False
            reward = 0.01

        # Fill background color
        DISPLAYSURF.fill(BLACK)

        # Display time
        self.time_msg("Survival Time: " + str(time.time() - self.start_time), (10, 15))

        # Draw agent
        pygame.draw.circle(DISPLAYSURF, BLUE, (int(self.my_position[0]), int(self.my_position[1])), self.my_radius, 0)

        # Draw ball
        for i in range(len(self.ball_list)):
            pygame.draw.circle(DISPLAYSURF, RED, (int(self.ball_list[i][1]), int(self.ball_list[i][2])), self.ball_radius, 0)

        # Draw lines for gameboard
        self.draw_board()

        pygame.display.update()
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        return image_data, reward, terminal

    # Exit the game
    def terminate(self):
    	pygame.quit()
    	sys.exit()

    # Set random position and velocity
    def set_ball_pos_and_vel(self):
        rand_pos_x = 0
        rand_pos_y = 0
        rand_vel_x = 0
        rand_vel_y = 0

        ball_list = []

        for i in range(self.num_balls):
            ball_list.append([])

            # Get random numbers
            rand_pos_x = random.random()
            rand_pos_y = random.random()
            rand_vel_x = random.random()
            rand_vel_y = random.random()

            ball_list[i].append(i) # id

            # initial x position
            if rand_pos_x > 0.5:
                ball_list[i].append(random.randint(CENTER_X + self.gap_balls, WINDOW_WIDTH - self.gap_balls))
            else:
                ball_list[i].append(random.randint(self.gap_balls, CENTER_X - self.gap_balls))

            # initial y position
            if rand_pos_y > 0.5:
                ball_list[i].append(random.randint(CENTER_Y + self.gap_balls, WINDOW_HEIGHT - self.gap_balls))
            else:
                ball_list[i].append(random.randint(TOP_WIDTH + self.gap_balls, CENTER_Y - self.gap_balls))

            # initial x velocity
            if rand_vel_x > 0.5:
                ball_list[i].append(random.uniform(self.min_ball_speed, self.max_ball_speed))
            else:
                ball_list[i].append(-random.uniform(self.min_ball_speed, self.max_ball_speed))

            # initial y velocity
            if rand_vel_y > 0.5:
                ball_list[i].append(random.uniform(self.min_ball_speed, self.max_ball_speed))
            else:
                ball_list[i].append(-random.uniform(self.min_ball_speed, self.max_ball_speed))

        return ball_list

    # Keep the agent inside gameboard
    def constraint(self):
        if self.my_position[0] <= GAP_WIDTH + self.my_radius:
            self.my_position[0] = GAP_WIDTH + self.my_radius

        if self.my_position[0] >= WINDOW_WIDTH - GAP_WIDTH - self.my_radius:
            self.my_position[0] = WINDOW_WIDTH - GAP_WIDTH - self.my_radius

        if self.my_position[1] >= WINDOW_HEIGHT - GAP_WIDTH - self.my_radius:
            self.my_position[1] = WINDOW_HEIGHT - GAP_WIDTH - self.my_radius

        if self.my_position[1] <= TOP_WIDTH + GAP_WIDTH + self.my_radius:
            self.my_position[1] = TOP_WIDTH + GAP_WIDTH + self.my_radius

    # Update balls
    def update_balls(self):
        for i in range(self.num_balls):
            # Move the balls
            self.ball_list[i][1] += self.ball_list[i][3]
            self.ball_list[i][2] += self.ball_list[i][4]

            # If ball hits the ball, it bounce
            if self.ball_list[i][1] <= GAP_WIDTH + self.ball_radius:
                self.ball_list[i][1] = GAP_WIDTH + self.ball_radius + 1
                self.ball_list[i][3] = -self.ball_list[i][3]

            if self.ball_list[i][1] >= WINDOW_WIDTH - GAP_WIDTH - self.ball_radius:
                self.ball_list[i][1] = WINDOW_WIDTH - GAP_WIDTH - self.ball_radius - 1
                self.ball_list[i][3] = -self.ball_list[i][3]

            if self.ball_list[i][2] >= WINDOW_HEIGHT - GAP_WIDTH - self.ball_radius:
                self.ball_list[i][2] = WINDOW_HEIGHT - GAP_WIDTH - self.ball_radius - 1
                self.ball_list[i][4] = -self.ball_list[i][4]

            if self.ball_list[i][2] <= TOP_WIDTH + GAP_WIDTH + self.ball_radius:
                self.ball_list[i][2] = TOP_WIDTH + GAP_WIDTH + self.ball_radius + 1
                self.ball_list[i][4] = -self.ball_list[i][4]

    # Check lose
    def check_lose(self):
        # check collision
        for i in range(self.num_balls):
            x_square = (self.my_position[0] - self.ball_list[i][1]) ** 2
            y_square = (self.my_position[1] - self.ball_list[i][2]) ** 2
            dist_balls = self.my_radius + self.ball_radius

            if (np.sqrt(x_square + y_square) < dist_balls):
                self.init =  True
                return True

        self.init = False
        return False

    # Display time
    def time_msg(self, survive_time, position):
    	timeSurf = BASIC_FONT.render(str(survive_time), True, WHITE)
    	timeRect = timeSurf.get_rect()
    	timeRect.topleft = position
    	DISPLAYSURF.blit(timeSurf, timeRect)

    # Draw gameboard
    def draw_board(self):
        pygame.draw.line(DISPLAYSURF, WHITE, (GAP_WIDTH, TOP_WIDTH + GAP_WIDTH), (GAP_WIDTH, WINDOW_HEIGHT - GAP_WIDTH), 3)
        pygame.draw.line(DISPLAYSURF, WHITE, (WINDOW_WIDTH - GAP_WIDTH, TOP_WIDTH + GAP_WIDTH), (WINDOW_WIDTH - GAP_WIDTH, WINDOW_HEIGHT - GAP_WIDTH), 3)
        pygame.draw.line(DISPLAYSURF, WHITE, (GAP_WIDTH, TOP_WIDTH + GAP_WIDTH), (WINDOW_WIDTH - GAP_WIDTH, TOP_WIDTH + GAP_WIDTH), 3)
        pygame.draw.line(DISPLAYSURF, WHITE, (GAP_WIDTH, WINDOW_HEIGHT - GAP_WIDTH), (WINDOW_WIDTH - GAP_WIDTH, WINDOW_HEIGHT - GAP_WIDTH), 3)

if __name__ == '__main__':
	main()

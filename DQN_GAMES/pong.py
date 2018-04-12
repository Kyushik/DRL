# Atari pong
# By KyushikMin kyushikmin@gamil.com
# http://mmc.hanyang.ac.kr

import random, sys, time, math, pygame
from pygame.locals import *
import numpy as np

# Window Information
WINDOW_WIDTH = 480
WINDOW_HEIGHT = 360

HALF_WINDOW_WIDTH = int(WINDOW_WIDTH / 2)
HALF_WINDOW_HEIGHT = int(WINDOW_HEIGHT / 2)

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
    return 'pong'

def Return_Num_Action():
    return 3

my_bar_width = 10
my_bar_height = 50
my_bar_init_position = (WINDOW_HEIGHT - my_bar_height)/2
my_bar_speed = 10

enemy_bar_width = 10
enemy_bar_height = 100
enemy_bar_init_position = (WINDOW_HEIGHT - enemy_bar_height)/2
enemy_bar_speed = 10

ball_init_position_x = WINDOW_WIDTH / 2
ball_init_position_y = WINDOW_HEIGHT / 2

ball_radius = 5

class GameState:
    def __init__(self):
        global DISPLAYSURF, BASIC_FONT

        pygame.init()

        DISPLAYSURF = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

        pygame.display.set_caption('Pong')
        BASIC_FONT = pygame.font.Font('freesansbold.ttf', 16)

        # Set initial parameters
        self.init = True
        self.my_score = 0
        self.enemy_score = 0
        self.hit_count = 0

    def frame_step(self, input): # Game loop
        # Initial settings
        if self.init == True:
            self.my_bar_position = my_bar_init_position
            self.enemy_bar_position = enemy_bar_init_position

            self.ball_position_x = ball_init_position_x
            self.ball_position_y = ball_init_position_y

            self.hit_count = 0

            random_start_x = random.randint(0, 1)
            random_start_y = random.randint(0, 1)

            if random_start_x == 0:
                self.ball_speed_x = - random.uniform(6.0, 9.0)
            else:
                self.ball_speed_x = random.uniform(6.0, 9.0)

            if random_start_y == 0:
                self.ball_speed_y = -random.uniform(6.0, 9.0)
            else:
                self.ball_speed_y = random.uniform(6.0, 9.0)

            self.init = False

        # Key settings
        for event in pygame.event.get(): # event loop
            if event.type == QUIT:
                self.terminate()

        # Control the bar
        if input[1] == 1:
            self.my_bar_position -= my_bar_speed
        elif input[2] == 1:
            self.my_bar_position += my_bar_speed

        # Constraint of the bar
        if self.my_bar_position <= 0:
            self.my_bar_position = 0

        if self.my_bar_position >= WINDOW_HEIGHT - my_bar_height:
            self.my_bar_position = WINDOW_HEIGHT - my_bar_height

        # Move the ball
        self.ball_position_x += self.ball_speed_x
        self.ball_position_y += self.ball_speed_y

        # Move the enemy
        self.enemy_bar_position = self.ball_position_y - (enemy_bar_height/2)

        # Constraint of enemy bar
        if self.enemy_bar_position <= 0:
            self.enemy_bar_position = 0

        if self.enemy_bar_position >= WINDOW_HEIGHT - enemy_bar_height:
            self.enemy_bar_position = WINDOW_HEIGHT - enemy_bar_height

        # Ball is bounced when the ball hit the wall
        if self.ball_position_y <= 0 or self.ball_position_y >= WINDOW_HEIGHT:
            self.ball_speed_y = - self.ball_speed_y

        reward = 0
        terminal = False

        # Ball is bounced when the ball hit the bar
        if self.ball_position_x <= my_bar_width:
            # Hit the ball!
            if self.ball_position_y <= self. my_bar_position + my_bar_height and self.ball_position_y >= self.my_bar_position:
                self.ball_position_x = my_bar_width + 1
                self.ball_speed_x = - self.ball_speed_x

                # When the ball is at the corner
                if self.ball_position_y >= WINDOW_HEIGHT:
                    self.ball_position_x = my_bar_width + 1
                    self.ball_position_y = WINDOW_HEIGHT -1
                    self.ball_speed_x = - self.ball_speed_x
                    self.ball_speed_y = - self.ball_speed_y

                if self.ball_position_y <= 0:
                    self.ball_position_x = my_bar_width +1
                    self.ball_position_y = 1
                    self.ball_speed_x = - self.ball_speed_x
                    self.ball_speed_y = - self.ball_speed_y

                reward = 1
                self.hit_count += 1

        # Lose :(
        if self.ball_position_x <= 0:
            self.enemy_score += 1

            if self.enemy_score > 10:
                self.enemy_score = 0
                self.my_score = 0

            reward = -1
            terminal = True
            self.init = True

        # The ball is bounced when enemy hit the ball
        if self.ball_position_x >= WINDOW_WIDTH - enemy_bar_width:
            # enemy hit the ball
            if self.ball_position_y <= self.enemy_bar_position + enemy_bar_height and self.ball_position_y >= self.enemy_bar_position:
                self.ball_position_x = WINDOW_WIDTH - enemy_bar_width - 1
                self.ball_speed_x = - self.ball_speed_x

                # When the ball is at the corner
                if self.ball_position_y >= WINDOW_HEIGHT:
                    self.ball_position_x = WINDOW_WIDTH - enemy_bar_width -1
                    self.ball_position_y = WINDOW_HEIGHT -1
                    self.ball_speed_x = - self.ball_speed_x
                    self.ball_speed_y = - self.ball_speed_y

                if self.ball_position_y <= 0:
                    self.ball_position_x = WINDOW_WIDTH - enemy_bar_width -1
                    self.ball_position_y = 1
                    self.ball_speed_x = - self.ball_speed_x
                    self.ball_speed_y = - self.ball_speed_y

        # WIN!! :)
        if self.ball_position_x >= WINDOW_WIDTH:
            self.my_score += 1

            if self.my_score > 10:
                self.enemy_score = 0
                self.my_score = 0

            reward = 1
            terminal = True
            self.init = True

        # If bar hit the ball more than hit count threshold, game is finished!
        if self.hit_count == 10:
            reward = 1
            terminal = True
            self.init = True

        # Fill background color
        DISPLAYSURF.fill(BLACK)

        # Display scores
        self.score_msg(self.my_score, ((WINDOW_WIDTH/2) - 45, (WINDOW_HEIGHT/2)-10))
        self.score_msg(self.enemy_score, ((WINDOW_WIDTH/2) + 35, (WINDOW_HEIGHT/2)-10))

        # Draw bar
        my_bar_rect = pygame.Rect(0, self.my_bar_position, my_bar_width, my_bar_height)
        pygame.draw.rect(DISPLAYSURF, RED, my_bar_rect)

        enemy_bar_rect = pygame.Rect(WINDOW_WIDTH - enemy_bar_width, self.enemy_bar_position, enemy_bar_width, enemy_bar_height)
        pygame.draw.rect(DISPLAYSURF, BLUE, enemy_bar_rect)

        # Draw ball
        pygame.draw.circle(DISPLAYSURF, WHITE, (int(self.ball_position_x), int(self.ball_position_y)), ball_radius, 0)

        # Draw line for seperate game and info
        pygame.draw.line(DISPLAYSURF, WHITE, (WINDOW_WIDTH/2, 0), (WINDOW_WIDTH/2, WINDOW_HEIGHT), 3)

        pygame.display.update()
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())

        return image_data, reward, terminal

    # Exit the game
    def terminate(self):
        pygame.quit()
        sys.exit()

    # Display score
    def score_msg(self, score, position):
        scoreSurf = BASIC_FONT.render(str(score), True, WHITE)
        scoreRect = scoreSurf.get_rect()
        scoreRect.topleft = position
        DISPLAYSURF.blit(scoreSurf, scoreRect)

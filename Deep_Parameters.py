# This is parameter setting for all deep learning algorithms
import sys
# Import games
sys.path.append("DQN_GAMES/")

# Action Num
import pong
import dot
import dot_test
import tetris
import wormy
import breakout as game

Gamma = 0.99
Learning_rate = 0.00025
Epsilon = 1
Final_epsilon = 0.1

Num_action = game.Return_Num_Action()

Num_replay_memory = 50000
Num_start_training = 50000
Num_training = 500000
Num_update = 5000
Num_batch = 32
Num_test = 250000
Num_skipFrame = 4
Num_stackFrame = 4
Num_colorChannel = 1

Num_plot_episode = 50

GPU_fraction = 0.2
Is_train = True
Load_path = ''

img_size = 80

first_conv   = [8,8,Num_colorChannel * Num_stackFrame,32]
second_conv  = [4,4,32,64]
third_conv   = [3,3,64,64]
first_dense  = [10*10*64, 512]
second_dense = [512, Num_action]

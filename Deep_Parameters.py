# This is parameter setting for all deep learning algorithms
import sys 
# Import games
sys.path.append("Wrapped_Game/")

# Action Num
# pong = 3
# dot, dot_test = 4
# tetris = 5
import pong as game
import dot  
import dot_test  
import tetris  
import wormy

Gamma = 0.99
Learning_rate = 0.00025
Epsilon = 1 
Final_epsilon = 0.1 

Num_action = game.Return_Num_Action()

# 30000 / 29900
Num_replay_memory = 20000
Num_start_training = 20000-1
Num_training = 100000
Num_update = 1000 - 1
Num_batch = 32
Num_test = 10000
Num_skipFrame = 4
Is_skip = True
Num_stackFrame = 4
Num_colorChannel = 1

Num_plot_episode = 20
Num_step_save = 50000

img_size = 80

first_conv   = [8,8,Num_colorChannel * Num_stackFrame,32]
second_conv  = [4,4,32,64]
third_conv   = [3,3,64,64]
first_dense  = [10*10*64, 1024]
second_dense = [1024, 256]
third_dense  = [256, Num_action]
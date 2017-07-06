# This is parameter setting for all deep learning algorithms
import sys 
# Import games
sys.path.append("Wrapped_Game/")

# Action Num
# pong = 3
# dot, dot_test = 4
# tetris = 5
import pong as game 
# import pong_test
import dot  
import dot_test 
import tetris  
import wormy

Gamma = 0.9
Learning_rate = 0.00025
Epsilon = 1 
Final_epsilon = 0.1 

Num_replay_memory = 20000
Num_start_training = 10000
Num_training = 50000
Num_update = 500
Num_batch = 32
Num_test = 10000
Num_skipFrame = 4
Num_stackFrame = 4
Num_colorChannel = 1

Num_plot_episode = 30
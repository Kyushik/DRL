# Import modules 
import sys 
import pygame
import tensorflow as tf 
import cv2
import random 
import numpy as np 
import copy 
import matplotlib.pyplot as plt 

# Import games
sys.path.append("Wrapped_Game/")
import dot as game 
import tetris 

# Parameter setting 
Num_action = 4
Gamma = 0.9
Learning_rate = 0.001 
Epsilon = 1 
Num_replay_memory = 50000
Num_training = 1000000
Num_update = 1000
Num_batch = 32

game_name = 'dot'

# Initialize weights and bias 
def weight_variable(shape):
    return tf.Variable(xavier_initializer(shape))

def bias_variable(shape):
	return tf.Variable(xavier_initializer(shape))

# Xavier Weights initializer
def xavier_initializer(shape):
	dim_sum = np.sum(shape)
	if len(shape) == 1:
		dim_sum += 1
	bound = np.sqrt(2.0 / dim_sum)
	return tf.random_uniform(shape, minval=-bound, maxval=bound)

# Convolution and pooling
def conv2d(x,w):
	return tf.nn.conv2d(x,w,strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def assign_network_to_target():
	update_wconv1 = tf.assign(w_conv1_target, w_conv1)
	update_wconv2 = tf.assign(w_conv2_target, w_conv2)
	update_wconv3 = tf.assign(w_conv3_target, w_conv3)
	update_bconv1 = tf.assign(b_conv1_target, b_conv1)
	update_bconv2 = tf.assign(b_conv2_target, b_conv2)
	update_bconv3 = tf.assign(b_conv3_target, b_conv3)
	update_wfc1   = tf.assign(w_fc1_target, w_fc1)
	update_wfc2   = tf.assign(w_fc2_target, w_fc2)
	update_bfc1 = tf.assign(b_fc1_target, b_fc1)
	update_bfc2 = tf.assign(b_fc2_target, b_fc2)
	
	sess.run(update_wconv1)
	sess.run(update_wconv2)
	sess.run(update_wconv3)
	sess.run(update_bconv1)
	sess.run(update_bconv2)
	sess.run(update_bconv3)
	sess.run(update_wfc1)
	sess.run(update_wfc2)
	sess.run(update_bfc1)
	sess.run(update_bfc2)

# Input 
x_image = tf.placeholder(tf.float32, shape = [None, 80, 80, 3])

# Convolution variables 
w_conv1 = weight_variable([3,3,3,8])
b_conv1 = bias_variable([8])

w_conv2 = weight_variable([3,3,8,16])
b_conv2 = bias_variable([16])

w_conv3 = weight_variable([3,3,16,32])
b_conv3 = bias_variable([32])

# Densely connect layer variables 
w_fc1 = weight_variable([10*10*32, 64])
b_fc1 = bias_variable([64])

w_fc2 = weight_variable([64, Num_action])
b_fc2 = bias_variable([Num_action])


# 
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

h_pool3_flat = tf.reshape(h_pool3, [-1, 10*10*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1)+b_fc1)

output = tf.nn.softmax(tf.matmul(h_fc1, w_fc2) + b_fc2)

# Convolution variables target
w_conv1_target = weight_variable([3,3,3,8])
b_conv1_target = bias_variable([8])

w_conv2_target = weight_variable([3,3,8,16])
b_conv2_target = bias_variable([16])

w_conv3_target = weight_variable([3,3,16,32])
b_conv3_target = bias_variable([32])

# Densely connect layer variables target
w_fc1_target = weight_variable([10*10*32, 64])
b_fc1_target = bias_variable([64])

w_fc2_target = weight_variable([64, Num_action])
b_fc2_target = bias_variable([Num_action])

# 
h_conv1_target = tf.nn.relu(conv2d(x_image, w_conv1_target) + b_conv1_target)
h_pool1_target = max_pool_2x2(h_conv1_target)

h_conv2_target = tf.nn.relu(conv2d(h_pool1, w_conv2_target) + b_conv2_target)
h_pool2_target = max_pool_2x2(h_conv2_target)

h_conv3_target = tf.nn.relu(conv2d(h_pool2, w_conv3_target) + b_conv3_target)
h_pool3_target = max_pool_2x2(h_conv3_target)

h_pool3_flat_target = tf.reshape(h_pool3_target, [-1, 10*10*32])
h_fc1_target = tf.nn.relu(tf.matmul(h_pool3_flat_target, w_fc1_target)+b_fc1_target)

output_target = tf.nn.softmax(tf.matmul(h_fc1_target, w_fc2_target) + b_fc2_target)

# Loss function and Train 
action_target = tf.placeholder(tf.float32, shape = [None, Num_action])
y_target = tf.reduce_sum(tf.mul(output, action_target), reduction_indices = 1)
y_prediction = tf.placeholder(tf.float32, shape = [None])

Loss = tf.reduce_mean(tf.square(y_target - y_prediction))
train_step = tf.train.AdamOptimizer(Learning_rate).minimize(Loss)

# Initialize variables
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

# Load the file if the saved file exists
saver = tf.train.Saver()
check_save = input('Is there any saved data?(1=y/2=n): ')

if check_save == 1:
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

# Initial parameters
Replay_memory = []
step = 1
state = 'Observing'
score = 0 
episode = 0

game_state = game.GameState()
action = np.zeros([Num_action])
observation, reward, terminal = game_state.frame_step(action)
observation = cv2.resize(observation, (80, 80))
# observation = np.reshape(observation, (1, 80, 80, 3))
observation = observation/255.0

# Making replay memory
for i in range(Num_replay_memory):
	action = np.zeros([Num_action])
	action[random.randint(0, Num_action - 1)] = 1.0
	observation_next, reward, terminal = game_state.frame_step(action)
	observation_next = cv2.resize(observation_next, (80, 80))
	# observation_next = np.reshape(observation_next, (1, 80, 80, 3))
	observation_next = observation_next/255.0

	Replay_memory.append([observation, action, reward, observation_next, terminal])
	observation = observation_next
	if step % 100 == 0:
		print('step: ' + str(step) + ' / '  + 'state: ' + state)
	step += 1

plt.figure(1)
while True:
	if step <= Num_replay_memory + Num_training:
		# Training 
		state = 'Training'
		del Replay_memory[0]

		# if random value(0 - 1) is smaller than Epsilon, action is random. Otherwise, action is the one which has the largest Q value 
		if random.random() < Epsilon:
			action = np.zeros([Num_action])
			action[random.randint(0, Num_action - 1)] = 1.0
			observation_next, reward, terminal = game_state.frame_step(action)	
			observation_next = cv2.resize(observation_next, (80, 80))	
			# observation_next = np.reshape(observation_next, (1, 80, 80, 3))
			observation_next = observation_next/255.0
		else:
			observation_feed = np.reshape(observation, (1, 80, 80, 3))
			Q_value = output.eval(feed_dict={x_image: observation_feed})
			action = np.zeros([Num_action])
			action[np.argmax(Q_value)] = 1.0
			observation_next, reward, terminal = game_state.frame_step(action)
			observation_next = cv2.resize(observation_next, (80, 80))
			# observation_next = np.reshape(observation_next, (1, 80, 80, 3))
			observation_next = observation_next/255.0

		# Save experience to the Replay memory 
		Replay_memory.append([observation, action, reward, observation_next, terminal])	
		Replay_memory_copy = copy.copy(Replay_memory)
		
		# Shuffle the minibatch and slice it according to the number of batch 
		random.shuffle(Replay_memory_copy)
		minibatch = Replay_memory_copy[:Num_batch]

		# Save the each batch data 
		observation_batch      = [batch[0] for batch in minibatch]
		action_batch           = [batch[1] for batch in minibatch]
		reward_batch           = [batch[2] for batch in minibatch]
		observation_next_batch = [batch[3] for batch in minibatch]
		terminal_batch 	       = [batch[4] for batch in minibatch]

		y_batch = [] 

		# Update target network according to the Num_update value 
		if step % Num_update == 0:
			assign_network_to_target()

		# Get y_prediction 
		Q_batch = output_target.eval(feed_dict = {x_image: observation_next_batch})
		for i in range(len(minibatch)):
			if terminal_batch[i] == True:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + Gamma * np.max(Q_batch[i,:]))

		train_step.run(feed_dict = {action_target: action_batch, y_prediction: y_batch, x_image: observation_batch})

	    # save progress every 10000 iterations
		if step % 10000 == 0:
			saver.save(sess, 'saved_networks/' + game_name)
			print('Model is saved!!!')

		# Update parameters at every iteration	
		Epsilon -= 1.0/Num_training
		observation = observation_next
		step += 1
		score += reward 

		if terminal == True:
			plt.xlabel('Episode')
			plt.ylabel('Score')
			plt.grid(True)

			plt.plot(episode, score, hold = True, marker = 'o', ms = 3)
			plt.draw()
			plt.pause(0.000001)

			print('step: ' + str(step) + ' / '  + 'state: ' + state  + ' / '  + 'epsilon: ' + str(Epsilon) + ' / '  + 'score: ' + str(score)) 

			score = 0
			episode += 1
		
		if step == Num_replay_memory + Num_training:
			plt.savefig('./Plot/' + 'DQN' + game_name + '.png')			
	
	if step > Num_replay_memory + Num_training:
		# Testing
		state = 'Testing'
		Q_value = output.eval(feed_dict={x_image: observation})
		action = np.zeros([Num_action])
		action[np.argmax(Q_value)] = 1
		observation_next, reward, terminal = game_state.frame_step(action)
		observation_next = cv2.resize(observation_next, (80, 80))
		observation_next = np.reshape(observation_next, (1, 80, 80, 3))
		observation_next = observation_next/255.0

		observation = observation_next
		step += 1

	



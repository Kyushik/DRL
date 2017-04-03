# Import modules 
import sys 
import pygame
import tensorflow as tf 
import cv2
import random 
import numpy as np 
import copy 
import matplotlib.pyplot as plt 
import datetime 
from pushbullet.pushbullet import PushBullet

# Import games
sys.path.append("Wrapped_Game/")
import dot
import dot_test as game 
import tetris 

# Parameter setting 
Num_action = 4
Gamma = 0.99
Learning_rate = 0.00025 
Epsilon = 1 
Final_epsilon = 0.1 

Num_replay_memory = 20000
Num_start_training = 20000
Num_training = 200000
Num_update = 1000
Num_batch = 128

game_name = 'dot_test'

apiKey = "o.EaKxqzWHIba2UEX7oQEmMetS3MAN4ctW"
p = PushBullet(apiKey)
# Get a list of devices
devices = p.getDevices()

# # Initialize weights and bias 
# def weight_variable(shape):
#     return tf.Variable(xavier_initializer(shape))

# def bias_variable(shape):
# 	return tf.Variable(xavier_initializer(shape))

# # Xavier Weights initializer
# def xavier_initializer(shape):
# 	dim_sum = np.sum(shape)
# 	if len(shape) == 1:
# 		dim_sum += 1
# 	bound = np.sqrt(2.0 / dim_sum)
# 	return tf.random_uniform(shape, minval=-bound, maxval=bound)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

# Convolution and pooling
def conv2d(x,w, stride):
	return tf.nn.conv2d(x,w,strides=[1, stride, stride, 1], padding='SAME')

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
	update_wfc3   = tf.assign(w_fc3_target, w_fc3)
	update_bfc1 = tf.assign(b_fc1_target, b_fc1)
	update_bfc2 = tf.assign(b_fc2_target, b_fc2)
	update_bfc3 = tf.assign(b_fc3_target, b_fc3)

	sess.run(update_wconv1)
	sess.run(update_wconv2)
	sess.run(update_wconv3)
	sess.run(update_bconv1)
	sess.run(update_bconv2)
	sess.run(update_bconv3)
	sess.run(update_wfc1)
	sess.run(update_wfc2)
	sess.run(update_wfc3)
	sess.run(update_bfc1)
	sess.run(update_bfc2)
	sess.run(update_bfc3)

# Input 
x_image = tf.placeholder(tf.float32, shape = [None, 80, 80, 3])

# Convolution variables 
w_conv1 = weight_variable([8,8,3,16])
b_conv1 = bias_variable([16])

w_conv2 = weight_variable([4,4,16,32])
b_conv2 = bias_variable([32])

w_conv3 = weight_variable([3,3,32,64])
b_conv3 = bias_variable([64])

# Densely connect layer variables 
w_fc1 = weight_variable([10*10*64, 512])
b_fc1 = bias_variable([512])

w_fc2 = weight_variable([512, 256])
b_fc2 = bias_variable([256])

w_fc3 = weight_variable([256, Num_action])
b_fc3 = bias_variable([Num_action])


# 
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1, 4) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_conv1, w_conv2, 2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)

h_conv3 = tf.nn.relu(conv2d(h_conv2, w_conv3, 1) + b_conv3)
# h_pool3 = max_pool_2x2(h_conv3)

h_pool3_flat = tf.reshape(h_conv3, [-1, 10*10*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1)+b_fc1)
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2)+b_fc2)

output = tf.matmul(h_fc2, w_fc3) + b_fc3

# Convolution variables target
w_conv1_target = weight_variable([8,8,3,16])
b_conv1_target = bias_variable([16])

w_conv2_target = weight_variable([4,4,16,32])
b_conv2_target = bias_variable([32])

w_conv3_target = weight_variable([3,3,32,64])
b_conv3_target = bias_variable([64])

# Densely connect layer variables target
w_fc1_target = weight_variable([10*10*64, 512])
b_fc1_target = bias_variable([512])

w_fc2_target = weight_variable([512, 256])
b_fc2_target = bias_variable([256])

w_fc3_target = weight_variable([256, Num_action])
b_fc3_target = bias_variable([Num_action])

# 
h_conv1_target = tf.nn.relu(conv2d(x_image, w_conv1_target, 4) + b_conv1_target)
# h_pool1_target = max_pool_2x2(h_conv1_target)

h_conv2_target = tf.nn.relu(conv2d(h_conv1_target, w_conv2_target, 2) + b_conv2_target)
# h_pool2_target = max_pool_2x2(h_conv2_target)

h_conv3_target = tf.nn.relu(conv2d(h_conv2_target, w_conv3_target, 1) + b_conv3_target)
# h_pool3_target = max_pool_2x2(h_conv3_target)

h_pool3_flat_target = tf.reshape(h_conv3_target, [-1, 10*10*64])
h_fc1_target = tf.nn.relu(tf.matmul(h_pool3_flat_target, w_fc1_target)+b_fc1_target)
h_fc2_target = tf.nn.relu(tf.matmul(h_fc1_target, w_fc2_target)+b_fc2_target)

output_target = tf.matmul(h_fc2_target, w_fc3_target) + b_fc3_target

# Loss function and Train 
action_target = tf.placeholder(tf.float32, shape = [None, Num_action])
y_prediction = tf.placeholder(tf.float32, shape = [None])

y_target = tf.reduce_sum(tf.mul(output, action_target), reduction_indices = 1)
Loss = tf.reduce_mean(tf.square(y_prediction - y_target))
train_step = tf.train.AdamOptimizer(Learning_rate).minimize(Loss)

# Initialize variables
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config=config)
init = tf.global_variables_initializer()
sess.run(init)


# Load the file if the saved file exists
saver = tf.train.Saver()
check_save = input('Is there any saved data?(1=y/2=n): ')

if check_save == 1:
    checkpoint = tf.train.get_checkpoint_state("saved_networks_DQN")
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
observation = (observation - (255.0 / 2)) / (255.0 / 2)

# Making replay memory
for i in range(Num_start_training):
	action = np.zeros([Num_action])
	action[random.randint(0, Num_action - 1)] = 1.0
	observation_next, reward, terminal = game_state.frame_step(action)
	observation_next = cv2.resize(observation_next, (80, 80))
	observation_next = (observation_next - (255.0 / 2)) / (255.0 / 2)

	Replay_memory.append([observation, action, reward, observation_next, terminal])
	observation = observation_next
	if step % 100 == 0:
		print('step: ' + str(step) + ' / '  + 'state: ' + state)
	step += 1

plt.figure(1)
plot_x = []
plot_y = []
while True:
	if step <= Num_start_training + Num_training:
		# Training 
		state = 'Training'

		if len(Replay_memory) > Num_replay_memory:
			del Replay_memory[0]

		# if random value(0 - 1) is smaller than Epsilon, action is random. Otherwise, action is the one which has the largest Q value 
		if random.random() < Epsilon:
			action = np.zeros([Num_action])
			action[random.randint(0, Num_action - 1)] = 1
			observation_next, reward, terminal = game_state.frame_step(action)	
			observation_next = cv2.resize(observation_next, (80, 80))	
			observation_next = (observation_next - (255.0 / 2)) / (255.0 / 2)
		else:
			observation_feed = np.reshape(observation, (1, 80, 80, 3))
			Q_value = output.eval(feed_dict={x_image: observation_feed})[0]
			action = np.zeros([Num_action])
			action[np.argmax(Q_value)] = 1
			observation_next, reward, terminal = game_state.frame_step(action)
			observation_next = cv2.resize(observation_next, (80, 80))
			observation_next = (observation_next - (255.0 / 2)) / (255.0 / 2)

		# Save experience to the Replay memory 
		Replay_memory.append([observation, action, reward, observation_next, terminal])	
		minibatch =  random.sample(Replay_memory, Num_batch)

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
				y_batch.append(reward_batch[i] + Gamma * np.max(Q_batch[i]))

		train_step.run(feed_dict = {action_target: action_batch, y_prediction: y_batch, x_image: observation_batch})

	    # save progress every 10000 iterations
		if step % 10000 == 0:
			saver.save(sess, 'saved_networks_DQN/' + game_name)
			print('Model is saved!!!')

		# Update parameters at every iteration	
		observation = observation_next

		if Epsilon > Final_epsilon:
			Epsilon -= 1.0/Num_training
		
		if step == Num_start_training + Num_training:
			plt.savefig('./Plot/' + ' DQN ' + game_name + '.png')		

			# Send a note to pushbullet 
			p.pushNote(devices[0]["iden"], 'DQN', 'DQN is done')	
	
	if step > Num_start_training + Num_training:
		# Testing
		state = 'Testing'
		Q_value = output.eval(feed_dict={x_image: observation_feed})[0]
		action = np.zeros([Num_action])
		action[np.argmax(Q_value)] = 1
		observation_next, reward, terminal = game_state.frame_step(action)
		observation_next = cv2.resize(observation_next, (80, 80))
		observation_next = np.reshape(observation_next, (1, 80, 80, 3))
		observation_next = (observation_next - (255.0 / 2)) / (255.0 / 2)

		observation_feed = observation_next
	
	step += 1
	score += reward 

	if terminal == True:
		print('step: ' + str(step) + ' / '  + 'state: ' + state  + ' / '  + 'epsilon: ' + str(Epsilon) + ' / '  + 'score: ' + str(score)) 

		plot_x.append(episode)
		plot_y.append(score)

		score = 0
		episode += 1

	if len(plot_x) == 100:
		plt.xlabel('Episode')
		plt.ylabel('Score')
		plt.title('Deep Q Learning')
		plt.grid(True)

		plt.plot(np.average(plot_x), np.average(plot_y), hold = True, marker = '*', ms = 5)
		plt.draw()
		plt.pause(0.000001)

		plot_x = []
		plot_y = [] 


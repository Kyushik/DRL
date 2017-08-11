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
# from pushbullet.pushbullet import PushBullet
import time 

# Parameter Setting
import Deep_Parameters
game = Deep_Parameters.game

Num_action = game.Return_Num_Action()

Gamma = Deep_Parameters.Gamma
Learning_rate = Deep_Parameters.Learning_rate
Epsilon = Deep_Parameters.Epsilon
Final_epsilon = Deep_Parameters.Final_epsilon

Num_replay_memory = Deep_Parameters.Num_replay_memory
Num_start_training = Deep_Parameters.Num_start_training
Num_training = Deep_Parameters.Num_training
Num_update = Deep_Parameters.Num_update
Num_batch = Deep_Parameters.Num_batch
Num_test = Deep_Parameters.Num_test
Num_skipFrame = Deep_Parameters.Num_skipFrame
Is_skip = Deep_Parameters.Is_skip
Num_stackFrame = Deep_Parameters.Num_stackFrame
Num_colorChannel = Deep_Parameters.Num_colorChannel

Num_plot_episode = Deep_Parameters.Num_plot_episode
Num_step_save = Deep_Parameters.Num_step_save

# Parametwrs for Network
img_size = Deep_Parameters.img_size

first_conv   = Deep_Parameters.first_conv
second_conv  = Deep_Parameters.second_conv
third_conv   = Deep_Parameters.third_conv
first_dense  = Deep_Parameters.first_dense
second_dense = Deep_Parameters.second_dense
third_dense  = Deep_Parameters.third_dense

game_name = game.ReturnName()

# apiKey = "o.EaKxqzWHIba2UEX7oQEmMetS3MAN4ctW"
# p = PushBullet(apiKey)
# Get a list of devices
# devices = p.getDevices()

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

# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev = 0.01)
#     return tf.Variable(initial)

# def bias_variable(shape):
#     initial = tf.constant(0.01, shape = shape)
#     return tf.Variable(initial)

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

def resize_and_norm(observation):
	observation_out = cv2.resize(observation, (img_size, img_size))
	if Num_colorChannel == 1:
		observation_out = cv2.cvtColor(observation_out, cv2.COLOR_BGR2GRAY)
		observation_out = np.reshape(observation_out, (img_size, img_size, 1))

	observation_out = (observation_out - (255.0 / 2)) / (255.0 / 2)
	return observation_out 

# Input 
x_image = tf.placeholder(tf.float32, shape = [None, img_size, img_size, Num_colorChannel * Num_stackFrame])

# Convolution variables 
w_conv1 = weight_variable(first_conv)
b_conv1 = bias_variable([first_conv[3]])

w_conv2 = weight_variable(second_conv)
b_conv2 = bias_variable([second_conv[3]])

w_conv3 = weight_variable(third_conv)
b_conv3 = bias_variable([third_conv[3]])

# Densely connect layer variables 
w_fc1 = weight_variable(first_dense)
b_fc1 = bias_variable([first_dense[1]])

w_fc2 = weight_variable(second_dense)
b_fc2 = bias_variable([second_dense[1]])

w_fc3 = weight_variable(third_dense)
b_fc3 = bias_variable([third_dense[1]])

# Network
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1, 4) + b_conv1)
h_conv2 = tf.nn.relu(conv2d(h_conv1, w_conv2, 2) + b_conv2)
h_conv3 = tf.nn.relu(conv2d(h_conv2, w_conv3, 1) + b_conv3)

h_pool3_flat = tf.reshape(h_conv3, [-1, first_dense[0]])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1)+b_fc1)
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2)+b_fc2)

output = tf.matmul(h_fc2, w_fc3) + b_fc3

# Convolution variables target
w_conv1_target = weight_variable(first_conv)
b_conv1_target = bias_variable([first_conv[3]])

w_conv2_target = weight_variable(second_conv)
b_conv2_target = bias_variable([second_conv[3]])

w_conv3_target = weight_variable(third_conv)
b_conv3_target = bias_variable([third_conv[3]])

# Densely connect layer variables target
w_fc1_target = weight_variable(first_dense)
b_fc1_target = bias_variable([first_dense[1]])

w_fc2_target = weight_variable(second_dense)
b_fc2_target = bias_variable([second_dense[1]])

w_fc3_target = weight_variable(third_dense)
b_fc3_target = bias_variable([third_dense[1]])

# Target Network 
h_conv1_target = tf.nn.relu(conv2d(x_image, w_conv1_target, 4) + b_conv1_target)
h_conv2_target = tf.nn.relu(conv2d(h_conv1_target, w_conv2_target, 2) + b_conv2_target)
h_conv3_target = tf.nn.relu(conv2d(h_conv2_target, w_conv3_target, 1) + b_conv3_target)

h_pool3_flat_target = tf.reshape(h_conv3_target, [-1, first_dense[0]])
h_fc1_target = tf.nn.relu(tf.matmul(h_pool3_flat_target, w_fc1_target)+b_fc1_target)
h_fc2_target = tf.nn.relu(tf.matmul(h_fc1_target, w_fc2_target)+b_fc2_target)

output_target = tf.matmul(h_fc2_target, w_fc3_target) + b_fc3_target

# Loss function and Train 
action_target = tf.placeholder(tf.float32, shape = [None, Num_action])
y_prediction = tf.placeholder(tf.float32, shape = [None])

y_target = tf.reduce_sum(tf.multiply(output, action_target), reduction_indices = 1)
Loss = tf.reduce_mean(tf.square(y_prediction - y_target))
# train_step = tf.train.RMSPropOptimizer(Learning_rate).minimize(Loss)
train_step = tf.train.AdamOptimizer(learning_rate = Learning_rate, epsilon = 1e-02).minimize(Loss)
# train_step = tf.train.RMSPropOptimizer(learning_rate = Learning_rate, momentum = 0.95).minimize(Loss)

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
datetime_now = str(datetime.date.today()) 
hour = str(datetime.datetime.now().hour)

game_state = game.GameState()
action = np.zeros([Num_action])
observation, reward, terminal = game_state.frame_step(action)
observation = resize_and_norm(observation)
observation_copy = copy.deepcopy(observation)

observation_in = np.zeros([img_size, img_size, Num_colorChannel * Num_stackFrame])
observation_next_in = np.zeros([img_size, img_size, Num_colorChannel * Num_stackFrame])

observation_set = []
for i in range(Num_skipFrame * Num_stackFrame):
	observation_set.append(observation_copy)

# Making replay memory
for i in range(Num_start_training):
	action = np.zeros([Num_action])
	action[random.randint(0, Num_action - 1)] = 1.0

	observation_next, reward, terminal = game_state.frame_step(action)
	observation_next = resize_and_norm(observation_next)

	observation_set.append(observation_next)

	observation_next_in = np.zeros((img_size, img_size, 1))

	# Stack the frame according to the number of skipping frame 	
	for stack_frame in range(Num_stackFrame):
		observation_next_in = np.insert(observation_next_in, [1], observation_set[-1 - (Num_skipFrame * stack_frame)], axis = 2)

	del observation_set[0]

	observation_next_in = np.delete(observation_next_in, [0], axis = 2)

	Replay_memory.append([observation_in, action, reward, observation_next_in, terminal])
	
	observation_in = observation_next_in
	
	if step % 100 == 0:
		print('step: ' + str(step) + ' / '  + 'state: ' + state)
	step += 1

# Figure and figure data setting
plt.figure(1)
plot_x = []
plot_y = []

test_score = []

# Training & Testing 
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
		else:
			observation_feed = np.reshape(observation_in, (1, img_size, img_size, Num_colorChannel * Num_stackFrame))
			Q_value = output.eval(feed_dict={x_image: observation_feed})
			action = np.zeros([Num_action])
			action[np.argmax(Q_value)] = 1

		observation_next, reward, terminal = game_state.frame_step(action)
		observation_next = resize_and_norm(observation_next)

		observation_set.append(observation_next)

		observation_next_in = np.zeros((img_size, img_size, 1))

		# Stack the frame according to the number of skipping frame 
		for stack_frame in range(Num_stackFrame):
			observation_next_in = np.insert(observation_next_in, [1], observation_set[-1 - (Num_skipFrame * stack_frame)], axis = 2)

		del observation_set[0]

		observation_next_in = np.delete(observation_next_in, [0], axis = 2)

		# Save experience to the Replay memory 
		Replay_memory.append([observation_in, action, reward, observation_next_in, terminal])	

		# Update parameters at every iteration	
		observation_in = observation_next_in

		if Epsilon > Final_epsilon:
			Epsilon -= 1.0/Num_training
		
		# Select minibatch
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
		if step % Num_step_save == 0:
			saver.save(sess, 'saved_networks_DQN/' + game_name)
			print('Model is saved!!!')

	if step > Num_start_training + Num_training:
		# Testing
		state = 'Testing'
		Epsilon = 0

		# Choose the action of testing state
		observation_feed = np.reshape(observation_in, (1, img_size, img_size, Num_colorChannel * Num_stackFrame))
		Q_value = output.eval(feed_dict={x_image: observation_feed})
		action = np.zeros([Num_action])
		action[np.argmax(Q_value)] = 1
			
		# Get game state
		observation_next, reward, terminal = game_state.frame_step(action)
		observation_next = resize_and_norm(observation_next)

		observation_set.append(observation_next)

		observation_next_in = np.zeros((img_size, img_size, 1))

		# Stack the frame according to the number of skipping frame 
		for stack_frame in range(Num_stackFrame):
			observation_next_in = np.insert(observation_next_in, [1], observation_set[-1 - (Num_skipFrame * stack_frame)], axis = 2)
			
		del observation_set[0]
		
		observation_next_in = np.delete(observation_next_in, [0], axis = 2)

		observation_in = observation_next_in 

	if step == Num_start_training + Num_training + Num_test:
		mean_score_test = np.average(test_score) 
		print(game_name + str(mean_score_test))
		plt.savefig('./Plot/' + datetime_now + '_' + hour + '_DQN_' + game_name + str(mean_score_test) + '.png')		

		# # Send a note to pushbullet 
		# p.pushNote(devices[0]["iden"], 'DQN', 'DQN is done')
		
		# Finish the Code 
		break	

	step += 1
	score += reward 

	if terminal == True:
		print('step: ' + str(step) + ' / '  + 'state: ' + state  + ' / '  + 'epsilon: ' + str(Epsilon) + ' / '  + 'score: ' + str(score)) 

		plot_x.append(episode)
		plot_y.append(score)

		if state == 'Testing':
			test_score.append(score)

		score = 0
		episode += 1


	if len(plot_x) % Num_plot_episode == 0 and len(plot_x) != 0:
		plt.xlabel('Episode')
		plt.ylabel('Score')
		plt.title('Deep Q Learning')
		plt.grid(True)

		plt.plot(np.average(plot_x), np.average(plot_y), hold = True, marker = '*', ms = 5)
		plt.draw()
		plt.pause(0.000001)

		plot_x = []
		plot_y = [] 
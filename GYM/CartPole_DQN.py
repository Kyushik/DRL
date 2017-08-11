# Cartpole 
# State  -> x, x_dot, theta, theta_dot
# Action -> force (+1, -1)

# Import modules 
import tensorflow as tf 
import random
import numpy as np 
import copy 
import matplotlib.pyplot as plt 
import datetime 
import gym

env = gym.make('CartPole-v0')
game_name = 'CartPole'
algorithm = 'DQN'

# Parameter setting 
Num_action = 2
Gamma = 0.99
Learning_rate = 0.00025 
Epsilon = 1 
Final_epsilon = 0.01 

Num_replay_memory = 10000
Num_start_training = 1000
Num_training = 15000
Num_testing  = 5000 
Num_update = 150
Num_batch = 32
Num_episode_plot = 10

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

# Assigning network variables to target network variables 
def assign_network_to_target():
	update_wfc1 = tf.assign(w_fc1_target, w_fc1)
	update_wfc2 = tf.assign(w_fc2_target, w_fc2)
	update_wfc3 = tf.assign(w_fc3_target, w_fc3)
	update_bfc1 = tf.assign(b_fc1_target, b_fc1)
	update_bfc2 = tf.assign(b_fc2_target, b_fc2)
	update_bfc3 = tf.assign(b_fc3_target, b_fc3)

	sess.run(update_wfc1)
	sess.run(update_wfc2)
	sess.run(update_wfc3)
	sess.run(update_bfc1)
	sess.run(update_bfc2)
	sess.run(update_bfc3)

# Input 
x = tf.placeholder(tf.float32, shape = [None, 4])

# Densely connect layer variables 
w_fc1 = weight_variable([4, 1024])
b_fc1 = bias_variable([1024])

w_fc2 = weight_variable([1024, 256])
b_fc2 = bias_variable([256])

w_fc3 = weight_variable([256, Num_action])
b_fc3 = bias_variable([Num_action])

h_fc1 = tf.nn.relu(tf.matmul(x, w_fc1)+b_fc1)
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2)+b_fc2)

output = tf.matmul(h_fc2, w_fc3) + b_fc3


# Densely connect layer variables target
w_fc1_target = weight_variable([4, 1024])
b_fc1_target = bias_variable([1024])

w_fc2_target = weight_variable([1024, 256])
b_fc2_target = bias_variable([256])

w_fc3_target = weight_variable([256, Num_action])
b_fc3_target = bias_variable([Num_action])

h_fc1_target = tf.nn.relu(tf.matmul(x, w_fc1_target)+b_fc1_target)
h_fc2_target = tf.nn.relu(tf.matmul(h_fc1_target, w_fc2_target)+b_fc2_target)

output_target = tf.matmul(h_fc2_target, w_fc3_target) + b_fc3_target

# Loss function and Train 
action_target = tf.placeholder(tf.float32, shape = [None, Num_action])
y_prediction = tf.placeholder(tf.float32, shape = [None])

y_target = tf.reduce_sum(tf.multiply(output, action_target), reduction_indices = 1)
Loss = tf.reduce_mean(tf.square(y_prediction - y_target))
train_step = tf.train.AdamOptimizer(Learning_rate).minimize(Loss)

# Initialize variables
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config=config)
init = tf.global_variables_initializer()
sess.run(init)

# Initial parameters
Replay_memory = []
step = 1
score = 0 
episode = 0

# datetime_now = str(datetime.date.today()) 
# hour = str(datetime.datetime.now().hour)
# minite = str(datetime.datetime.now().minute)
data_time = str(datetime.date.today()) + '_' + str(datetime.datetime.now().hour) + '_' + str(datetime.datetime.now().minute)


observation = env.reset()
action = env.action_space.sample()
observation, reward, terminal, info = env.step(action)

# Figure and figure data setting
plt.figure(1)
plot_x = []
plot_y = []

# Making replay memory
while True:
	# Rendering
	env.render()

	if step <= Num_start_training:
		state = 'Observing'
		action = np.zeros([Num_action])
		action[random.randint(0, Num_action - 1)] = 1.0
		action_step = np.argmax(action)

		observation_next, reward, terminal, info = env.step(action_step)
		reward -= 5 * abs(observation_next[0])

		Replay_memory.append([observation, action, reward, observation_next, terminal])
		
		observation = observation_next
		
		if step % 10 == 0:
			print('step: ' + str(step) + ' / '  + 'state: ' + state)
		step += 1

	elif step <= Num_start_training + Num_training:
		# Training 
		state = 'Training'

		if len(Replay_memory) > Num_replay_memory:
			del Replay_memory[0]

		# if random value(0 - 1) is smaller than Epsilon, action is random. Otherwise, action is the one which has the largest Q value 
		if random.random() < Epsilon:
			action = np.zeros([Num_action])
			action[random.randint(0, Num_action - 1)] = 1.0
			action_step = np.argmax(action)
				
		else:
			observation_feed = np.reshape(observation, (1,4))
			Q_value = output.eval(feed_dict={x: observation_feed})[0]
			action = np.zeros([Num_action])
			action[np.argmax(Q_value)] = 1
			action_step = np.argmax(action)
		
		observation_next, reward, terminal, info = env.step(action_step)
		reward -= 5 * abs(observation_next[0])

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
		Q_batch = output_target.eval(feed_dict = {x: observation_next_batch})
		for i in range(len(minibatch)):
			if terminal_batch[i] == True:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + Gamma * np.max(Q_batch[i]))

		train_step.run(feed_dict = {action_target: action_batch, y_prediction: y_batch, x: observation_batch})

		# Update parameters at every iteration	
		observation = observation_next

		if Epsilon > Final_epsilon:
			Epsilon -= 1.0/Num_training
		
	elif step < Num_start_training + Num_training + Num_testing:
		# Testing
		state = 'Testing'
		observation_feed = np.reshape(observation, (1,4))
		Q_value = output.eval(feed_dict={x: observation_feed})[0]

		action = np.zeros([Num_action])
		action[np.argmax(Q_value)] = 1
		action_step = np.argmax(action)
		
		observation_next, reward, terminal, info = env.step(action_step)
		observation = observation_next

		Epsilon = 0

	else: 
		# Test is finished
		print('Test is finished!!')
		plt.savefig('./Plot/' + data_time + '_' + algorithm + '_' + game_name + '.png')	
		break

	step += 1
	score += reward 

	# Plot average score
	if len(plot_x) % Num_episode_plot == 0 and len(plot_x) != 0:
		plt.xlabel('Episode')
		plt.ylabel('Score')
		plt.title('Cartpole_DQN')
		plt.grid(True)

		plt.plot(np.average(plot_x), np.average(plot_y), hold = True, marker = '*', ms = 5)
		plt.draw()
		plt.pause(0.000001)

		plot_x = []
		plot_y = [] 

	# Terminal
	if terminal == True:
		print('step: ' + str(step) + ' / '  + 'state: ' + state  + ' / '  + 'epsilon: ' + str(Epsilon) + ' / '  + 'score: ' + str(score)) 

		# Plotting data
		plot_x.append(episode)
		plot_y.append(score)

		score = 0
		episode += 1
		observation = env.reset()



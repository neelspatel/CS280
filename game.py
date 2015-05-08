import numpy as np
import math
from random import randint, random
import csv
import pylab
import matplotlib.pyplot as plt
import os
from scipy.interpolate import UnivariateSpline

#class for a game

#states are (row, column) tuples
#this function determines whether a position is valid given a board size
def is_valid_position(row,column,number_rows,number_columns):
	return (row < number_rows and column < number_columns)

#generates a list of all possible states
def all_possible_states(number_rows,number_columns):
	possible_states = []

	for i in range(number_rows):
		for j in range(number_columns):
			possible_states.append((i,j))

	return possible_states

#generates a list of valid neighboring states
def get_neighboring_states(row,column,all_possible_states):
	offsets = [(0,1),(0,-1),(-1,0),(1,0)]
	valid_neighbors = []

	for offset in offsets:
		offset_coordinate = (row+offset[0], column+offset[1])

		if offset_coordinate in all_possible_states:
			valid_neighbors.append(offset_coordinate)

	return valid_neighbors	

def get_next_state(row, column, action_index):
	if action_index==0: #up
		return (row-1, column)
	elif action_index==1: #right
		return (row, column+1)
	elif action_index==2: #down
		return (row+1, column)
	elif action_index==3: #left
		return (row, column-1)
	else:
		return False

#function for value iteration
#rows: number of rows in the game board
#columns: number of columns in the game board
#goal_state: index of the goal state
#horizon: number of iterations to go through
def value_iteration(rows, columns, goal_state, horizon=200):
	num_states = rows * columns
	all_states = all_possible_states(rows, columns)

	#initializes reward matrix appropriately
	values = [[0 for i in range(num_states)] for j in range(horizon)]
	for i in range(horizon):
		values[i][goal_state] = rows * columns

	q = [[0,0,0,0] for i in range(num_states)]

	for iteration in range(horizon):
		for cur_row in range(rows):
			for cur_column in range(columns):
				cur_state_index = cur_row * columns + cur_column
				#actions are of the form [up, right, down, left]
				for action in range(4):
					#reward + for all possible neighboring states, the previous value of that state
					next_state = get_next_state(cur_row, cur_column, action)
					
					if next_state:
						next_state_row, next_state_col = next_state

						#if this is a valid move
						if next_state_row >=0 and next_state_row < rows and next_state_col >=0 and next_state_col < columns:							
							next_state_index = next_state_row * columns + next_state_col

							#if next_state_index==goal_state:
							if cur_state_index==goal_state:
								reward = rows * columns
							else:
								reward = -1

							q[cur_state_index][action] = reward
							q[cur_state_index][action] += values[iteration-1][next_state_index]

				#determines the optimal action
				optimal_action = np.argmax(q[cur_state_index])		
				values[iteration][cur_state_index] = q[cur_state_index][optimal_action]

	return values, q

#given a current state, a goal state, and a q function,
#determines the optimal next action
def get_best_move(cur_row, cur_col, goal_row, goal_column, rows, columns, q):	
	#if any states are unreachable, mark them 
	unreachable_states = []

	#can't go up
	if cur_row == 0:
		unreachable_states.append(0)
	#can't go down
	elif cur_row == rows-1:
		unreachable_states.append(2)

	#can't go left
	if cur_col == 0:
		unreachable_states.append(3)
	#can't go right
	elif cur_col == columns-1:
		unreachable_states.append(1)

	cur_state_index = cur_row * columns + cur_col

	optimal_indices = np.argsort(q[cur_state_index])
	allowed_indices = [x for x in optimal_indices if x not in unreachable_states]
	optimal_index = allowed_indices[-1]

	#print q[cur_state_index], optimal_indices, allowed_indices, optimal_index

	return optimal_index

def at_goal(start_row, start_col, goal_row, goal_column):
	if start_row==goal_row and start_col == goal_column:		
		return True
	return False

def move(start_row, start_col, goal_row, goal_column, rows, columns, q, types, steps=0, output_file=None):
	next_row, next_col = start_row, start_col
	#checks if we have made it to the goal
	print "Started at (%d, %d)" % (start_row, start_col)

	history = []

	while not at_goal(next_row, next_col, goal_row, goal_column):
		
		best_move = get_best_move(next_row, next_col, goal_row, goal_column, rows, columns, q)
		history.append((next_row, next_col, best_move))

		next_row, next_col = get_next_state(next_row, next_col, best_move)


		print "Moved to (%d, %d)" % (next_row, next_col)		
		steps += 1

		#now simulates an observing agent
		probability_estimates = []
		for agent_type in types:
			probability = estimate_type_probability(history, agent_type["q"], columns, len(types))
			#print agent_type["label"], probability

			probability_estimates.append((agent_type["label"], probability))
		

		total_probability = float(sum(x[1] for x in probability_estimates))
		normalized_probability = [(label, x/total_probability) for (label, x) in probability_estimates]

		for label, probability in normalized_probability:
			print label, probability

			if output_file:
				output_file.write(str(probability)+"\t")

		if output_file:
			output_file.write("\n")

	print "Made it to the goal!"
	return steps, history

#adds an observation to the appropriate history based on the scheme
def assign_observation(observation, histories, observation_scheme):
	#if we are rotating through, find the first history with the fewest 
	#observations and add to that, or add to the first history
	if observation_scheme == "rotate":
		assign_index = 0

		#checks through every index in history, finding the first index
		#which has fewer entries than the previous index
		for i in range(1, len(histories)):
			if len(histories[i]) < len(histories[i-1]):
				assign_index = i
				break

		histories[assign_index].append(observation)

	#if we are choosing one randomly, choose a random index to assign
	#this observation to
	elif observation_scheme == "random_one":
		assign_index = randint(0,len(histories)-1)
		histories[assign_index].append(observation)

	#if we are choosing all observers randomly, then for each observer,
	#assign this observation with probability 1/n for n observers
	elif observation_scheme == "random_all":
		for i in range(len(histories)):
			use_this = random()

			if use_this > 1.0 / len(histories):
				histories[i].append(observation)

	#otherwise, assume that observation_scheme is "all"
	else:
		for i in range(len(histories)):
			histories[i].append(observation)


#given a list of probability estimates, where each item in the list is an array
#of probability estimates for each type, returns the average probability of each type
def get_average_estimates(probability_estimates):
	num_types = len(probability_estimates[0])
	num_estimates = float(len(probability_estimates))

	#probability_sums = [sum([x[i] for x in probability_estimates]) for i in range(num_types)]
	#probability_averages = [x/num_estimates for x in probability_sums]

	total_probability = {}

	for estimate in probability_estimates:
		for (label, probability) in estimate:
			if label not in total_probability:
				total_probability[label] = 0

			total_probability[label] += probability

	probability_averages = [(x, total_probability[x]/num_estimates) for x in total_probability.keys()]

	#sorts the averages by key
	probability_averages = sorted(probability_averages, key=lambda x: x[0])

	return probability_averages


def move_multiple_observers(start_row, start_col, goal_row, goal_column, rows, columns, q, types, steps=0, observers=1, observation_scheme="rotate", output_file=None):
	next_row, next_col = start_row, start_col
	#checks if we have made it to the goal
	print "Started at (%d, %d)" % (start_row, start_col)

	histories = [[] for i in range(observers)]

	while not at_goal(next_row, next_col, goal_row, goal_column):
		
		best_move = get_best_move(next_row, next_col, goal_row, goal_column, rows, columns, q)
		#history.append((next_row, next_col, best_move))
		assign_observation((next_row, next_col, best_move), histories, observation_scheme)

		next_row, next_col = get_next_state(next_row, next_col, best_move)


		print "Moved to (%d, %d)" % (next_row, next_col)		
		steps += 1

		#now simulates an observing agent
		probability_estimates = [[] for i in range(observers)]

		for i in range(observers):
			current_probability_estimate = []

			for agent_type in types:
				probability = estimate_type_probability(histories[i], agent_type["q"], columns, len(types))
				#print agent_type["label"], probability

				current_probability_estimate.append((agent_type["label"], probability))
			

			total_probability = float(sum(x[1] for x in current_probability_estimate))
			normalized_probability = [(label, x/total_probability) for (label, x) in current_probability_estimate]

			probability_estimates[i] = normalized_probability
		
		#combines the probability estimates from all observers
		averaged_estimates = get_average_estimates(probability_estimates)

		print averaged_estimates

		for label, probability in normalized_probability:
			#print label, probability

			if output_file:
				output_file.write(str(probability)+"\t")

		if output_file:
			output_file.write("\n")

	print "Made it to the goal!"
	return steps, histories


#to estimate type:
#for each type
#	for every goal with a nonzero probability for that type, sum += 
#		(term based on q function) * p(g|T) * p(T)
#		(term based on q function) = 1 * for each observation in history:
#			if the observation has state s and action a,
#			e^(B*Q(s,a))

def likelihood(weights, index):
	list_min = min(weights)
	list_max = max(weights)

	list_diff = float(list_max - list_min)
	if list_diff == 0:
		list_diff = 1

	new_values = [(i - list_min)/float(list_diff) for i in weights]

	

	try:
		return new_values[index] / sum(new_values)
	except ZeroDivisionError:
		return 0

def softmax(probabilities, index):
	prob_sum = float(sum(probabilities))
	relative_probs = [x/prob_sum for x in probabilities]

	exp_probs = [math.pow(math.e, x) for x in relative_probs]
	exp_sum = float(sum(exp_probs))
	relative_exp_probs = [x/exp_sum for x in exp_probs]	

	return relative_exp_probs[index]

#history is a list of (state_row, state_col, action) tuples
def estimate_type_probability(history, q, columns, num_types=2):
	probability = 0
	
	p_history_given_goal = 1
	for state_row, state_col, action in history:
		state_index = state_row * columns + state_col

		#p_history_given_goal += math.pow(math.e, q[state_index][action])
		#p_history_given_goal *= q[state_index][action] / float(sum(q[state_index]))
		#p_history_given_goal *= likelihood(q[state_index],action)
		p_history_given_goal *= math.pow(math.e, likelihood(q[state_index],action))
		#p_history_given_goal *= softmax(q[state_index],action)
		

	p_goal_given_type = 1 
	p_type = 1.0 / num_types

	#probability += math.pow(math.e,p_history_given_goal) / p_goal_given_type * p_type
	probability += p_history_given_goal / p_goal_given_type * p_type

	return probability

#actions are of the form [up, right, down, left]
def test(rows, columns, num_goals, output_file_name, iterations=1, horizon=200):
	for i in range(iterations):
		output_file = open(output_file_name + str(i) + ".txt", "w+")

		#generates a random index for each possible goal
		goal_indices = [randint(0,rows*columns-1) for i in range(num_goals)]
		goals = []

		start_index = randint(0,rows*columns-1)
		start_row = start_index/columns
		start_col = start_index%columns

		print "Starting at (%d,%d)" % (start_row, start_col)

		for counter, goal_index in enumerate(goal_indices):
			cur_goal = {}
			cur_goal['row'] = goal_index/columns
			cur_goal['column'] = goal_index%columns
			values, q = value_iteration(rows, columns, goal_index, rows*columns)
			cur_goal['q'] = q
			cur_goal['label'] = str(counter)

			print "Goal %d at (%d,%d)" % (counter, cur_goal['row'], cur_goal['column'])

			goals.append(cur_goal)
		
		true_goal = goals[0]
		true_goal_row = true_goal['row']
		true_goal_col = true_goal['column']
		true_goal_q = true_goal['q']

		print true_goal_q

		move(start_row, start_col, true_goal_row, true_goal_col, rows, columns, true_goal_q, goals, output_file=output_file)

		output_file.close()

def test_fixed_goals(rows, columns, num_goals, output_file_name, iterations=1, horizon=200):
	for i in range(iterations):
		output_file = open(output_file_name + str(i) + ".txt", "w+")

		#generates a random index for each possible goal
		#goal_indices = [randint(0,rows*columns-1) for i in range(num_goals)]
		goal_indices = [1,rows*columns-2]
		#goal_indices = [1,rows*columns-2,columns * (rows-1), columns-1]
		goals = []

		start_index = randint(0,rows*columns-1)
		start_row = start_index/columns
		start_col = start_index%columns

		print "Starting at (%d,%d)" % (start_row, start_col)

		for counter, goal_index in enumerate(goal_indices):
			cur_goal = {}
			cur_goal['row'] = goal_index/columns
			cur_goal['column'] = goal_index%columns
			values, q = value_iteration(rows, columns, goal_index, rows*columns)
			cur_goal['q'] = q
			cur_goal['label'] = str(counter)

			print "Goal %d at (%d,%d)" % (counter, cur_goal['row'], cur_goal['column'])

			goals.append(cur_goal)
		
		true_goal = goals[0]
		true_goal_row = true_goal['row']
		true_goal_col = true_goal['column']
		true_goal_q = true_goal['q']

		print true_goal_q

		move(start_row, start_col, true_goal_row, true_goal_col, rows, columns, true_goal_q, goals, output_file=output_file)

		output_file.close() 

def test_fixed_goals_multiple_observers(rows, columns, num_goals, output_file_name, observers=1, observation_scheme="all", iterations=1, horizon=200):
	for i in range(iterations):
		output_file = open(output_file_name + str(i) + ".txt", "w+")

		#generates a random index for each possible goal
		#goal_indices = [randint(0,rows*columns-1) for i in range(num_goals)]
		goal_indices = [1,rows*columns-2]
		#goal_indices = [1,rows*columns-2,columns * (rows-1), columns-1]
		goals = []

		start_index = randint(0,rows*columns-1)
		start_row = start_index/columns
		start_col = start_index%columns

		print "Starting at (%d,%d)" % (start_row, start_col)

		for counter, goal_index in enumerate(goal_indices):
			cur_goal = {}
			cur_goal['row'] = goal_index/columns
			cur_goal['column'] = goal_index%columns
			values, q = value_iteration(rows, columns, goal_index, rows*columns)
			cur_goal['q'] = q
			cur_goal['label'] = str(counter)

			print "Goal %d at (%d,%d)" % (counter, cur_goal['row'], cur_goal['column'])

			goals.append(cur_goal)
		
		true_goal = goals[0]
		true_goal_row = true_goal['row']
		true_goal_col = true_goal['column']
		true_goal_q = true_goal['q']

		print true_goal_q

		move_multiple_observers(start_row, start_col, true_goal_row, true_goal_col, rows, columns, true_goal_q, goals, observers=observers, observation_scheme=observation_scheme, output_file=output_file)

		output_file.close() 

def run_tests():
	observation_schemes = ["rotate", "random_one", "random_all"]
	for observers in range(2,7):
		for scheme in observation_schemes:
			game.test_fixed_goals_multiple_observers(10,10,2,"normalized_prob/10_10_2_fixed_"+str(observers)+"_"+scheme+"/test",observers=observers,observation_scheme=scheme,iterations=50)

def relative_probability(probabilities):
	prob_sum = float(sum(probabilities))
	relative_probs = [x/prob_sum for x in probabilities]

	exp_probs = [math.pow(math.e, x) for x in relative_probs]
	exp_sum = float(sum(exp_probs))
	relative_exp_probs = [x/exp_sum for x in exp_probs]

	print relative_exp_probs

	return relative_exp_probs


def scale(data, number):
	x = []
	y = []
	for i in range(len(data)):
		new_index = int(number * i / float(len(data)))
		x.append(new_index)
		y.append(data[i])

	return x,y

def read_data(filename):
	data = []
	with open(filename, "rb") as inputfile:
		print filename
		reader = csv.reader(inputfile, delimiter="\t")
		for row in reader:
			row = row[:-1]
			if len(row) > 1:
				data.append(map(float, row))

	return data

def graph(filename):
	data = read_data(filename)	

	#calculates the relative probability at each step
	probabilities = map(relative_probability, data)

	#scales the data (which may be an arbitrary number of steps) across n steps
	x,y = scale(probabilities, 100)

	#graphs data visually
	fig = plt.figure(figsize=(8,6))

	#for the number of types in the y data
	for i in range(len(y[0])):
		current_y = [current[i] for current in y]
		try:
			plt.plot(x, current_y, label=str(i), lw=2, marker='o')
		except e:
			print i, e

	plt.show()

	return x,y

def generate_graph(rows, cols, goals, fixed, observers, observation_scheme):
	observation_scheme_string = ""

	if observation_scheme == "random_all":
		observation_scheme_string += " using the RandomAll scheme"
	elif observation_scheme == "random_one":
		observation_scheme_string += " using the RandomOne scheme"
	elif observation_scheme == "rotate":
		observation_scheme_string += " using the Rotate scheme"

	title = "%d x %d grid with %d goals, %d observers%s" % (rows, cols, goals, observers, observation_scheme_string)

	if observers <= 1:
		folder_name = "normalized_prob/%d_%d_%d_%s" % (rows, cols, goals, fixed)
	else:
		folder_name = "normalized_prob/%d_%d_%d_%s_%d_%s" % (rows, cols, goals, fixed, observers, observation_scheme)

	easy_graph(folder_name, title)



def easy_graph(directory, title="Figure Title"):
	files = os.listdir(directory)

	if ".DS_Store" in files:
		files.remove(".DS_Store")

	filenames = [directory+"/"+x for x in files]
	graph_multiple(filenames, title)

def easy_compare(directories):
	fig = plt.figure(figsize=(10,4))

	plt.suptitle("Type Probabilities for Correct Type")

	colors = ["g", "r", "b", "c", "m"]

	lines = []
	line_labels = []

	for i, directory in enumerate(directories):
		files = os.listdir(directory)

		if ".DS_Store" in files:
			files.remove(".DS_Store")

		filenames = [directory+"/"+x for x in files]
		splines = get_splines(filenames)

		xnew = np.arange(0,100,0.1)
		ynew = splines[0](xnew)
		newline, = plt.plot(xnew, ynew, colors[i])
		pylab.ylim([-0.1,1.1])

		lines.append(newline)
		line_labels.append(directory)

	plt.legend(tuple(lines), tuple(line_labels), "lower right")

	plt.gca().tick_params(axis='x', labelsize=8)
	labels = plt.gca().axes.get_xticklabels()
	labels = ['' for x in labels]
	labels[0] = "Game start"
	labels[-1] = "Game end"
	plt.gca().axes.set_xticklabels(labels)

	plt.show(block=False)

def graph_multiple(filenames, title="Figure Title"):
	#graphs data visually
	fig = plt.figure(figsize=(10,4))

	plt.suptitle(title)

	colors = ["g", "r", "b", "c", "m"]

	all_data = [[] for i in range(4)]

	num_types = 2

	for filename in filenames:
		data = read_data(filename)	

		#calculates the relative probability at each step
		#probabilities = map(relative_probability, data)
		probabilities = data

		#scales the data (which may be an arbitrary number of steps) across n steps
		x,y = scale(probabilities, 100)
		
			
		#for the number of types in the y data
		if len(y) > 0:
			
			num_types = len(y[0])

			for i in range(len(y[0])):				
				current_y = [current[i] for current in y]
				current_unnormalized_y = [current[i] for current in probabilities]

				all_data[i] += zip(x, current_y)

				plt.subplot(1,3,1)
				plt.plot(range(1, len(current_unnormalized_y)+1), current_unnormalized_y, label=str(i), marker='o', color=colors[i])

				pylab.ylim([-0.1,1.1])

				plt.subplot(1,3,2)
				plt.plot(x, current_y, label=str(i), marker='o', color=colors[i])

				pylab.ylim([-0.1,1.1])

	#graphs the smooth spline

	#sorts each point in the set of all points by x value
	for i in range(num_types):
		all_data[i] = sorted(all_data[i], key= lambda x: x[0])

		all_x = [x[0] for x in all_data[i]]
		all_y = [x[1] for x in all_data[i]]

		spline = UnivariateSpline(all_x, all_y, k=4)
		xnew = np.arange(0,100,0.1)
		ynew = spline(xnew)

		plt.subplot(1,3,3)
		plt.plot(xnew, ynew, colors[i], linewidth=3)

		pylab.ylim([-0.1,1.1])

	#sets the titles and axes
	plt.subplot(1,3,1)	
	plt.gca().tick_params(axis='x', labelsize=8)
	plt.title("Type Probability By Move", fontsize=12)

	plt.subplot(1,3,2)	
	plt.gca().tick_params(axis='x', labelsize=8)
	labels = plt.gca().axes.get_xticklabels()
	labels = ['' for x in labels]
	labels[0] = "Game start"
	labels[-1] = "Game end"
	plt.gca().axes.set_xticklabels(labels)
	plt.title("Type Probability for Normalized Game", fontsize=12)

	plt.subplot(1,3,3)	
	plt.gca().tick_params(axis='x', labelsize=8)
	labels = plt.gca().axes.get_xticklabels()
	labels = ['' for x in labels]
	labels[0] = "Game start"
	labels[-1] = "Game end"
	plt.gca().axes.set_xticklabels(labels)
	plt.title("Smooth Type Probability", fontsize=12)


	plt.subplots_adjust(left=0.1, right=0.9, wspace=0.3, top=.8)

	plt.show(block=False)

	return x,y

def get_splines(filenames):
	all_data = [[] for i in range(4)]
	num_types = 2

	for filename in filenames:
		data = read_data(filename)	

		#calculates the relative probability at each step
		#probabilities = map(relative_probability, data)
		probabilities = data

		#scales the data (which may be an arbitrary number of steps) across n steps
		x,y = scale(probabilities, 100)
		
			
		#for the number of types in the y data
		if len(y) > 0:
			num_types = len(y[0])

			for i in range(len(y[0])):				
				current_y = [current[i] for current in y]
				all_data[i] += zip(x, current_y)

	splines = []

	#sorts each point in the set of all points by x value
	for i in range(num_types):
		all_data[i] = sorted(all_data[i], key= lambda x: x[0])

		all_x = [x[0] for x in all_data[i]]
		all_y = [x[1] for x in all_data[i]]

		spline = UnivariateSpline(all_x, all_y, k=4)

		splines.append(spline)
	
	return splines
	

#creates a goal at the top left
#v_left, q_left = game.value_iteration(5, 5, 0,300)
#left_goal = {"q": q_left, "row": 0, "column": 0, "label": "left type"}
#v_right, q_right = game.value_iteration(5, 5, 24,300)
#right_goal = {"q": q_right, "row": 4, "column": 4, "label": "right type"}
#types = [left_goal, right_goal]



import numpy as np
import random
import requests
import json
import subprocess
import sys
import signal
import sys
import pandas as pd
from sklearn.mixture import GaussianMixture
import numpy as np
import time
from collections import deque
from itertools import permutations
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Reshape, Input
from tensorflow.keras.optimizers import Adam

import nginxparser as ng

# change all layer default output to float64; default is float32
tf.keras.backend.set_floatx('float64')

'''generate possible weight combinations'''


def gen_control_list():
	NUM_CLUS = 3
	SRV_PER_CLUS = 3
	CONTROL_VALS = [1, 3, 5] * NUM_CLUS
    POSSIBLE_WEIGHTS = []
    perm = permutations(CONTROL_VALS, NUM_CLUS * SRV_PER_CLUS)
    tmp = []
    for c in list(perm):
        tmp.append(c)
    global POSSIBLE_WEIGHTS
    for elem in tmp:
        if (elem[0] / elem[-1]) % 1 != 0 or (elem[1] / elem[-1]) % 1 != 0 or (elem[2] / elem[-1]) % 1 != 0:
            POSSIBLE_WEIGHTS.append(elem)
        else:
            POSSIBLE_WEIGHTS.append((
                (int)(elem[0] / elem[-1]), (int)(elem[1] / elem[-1]), (int)(elem[2] / elem[-1])))
    s = set(POSSIBLE_WEIGHTS)
    POSSIBLE_WEIGHTS = list(s)

    for w in POSSIBLE_WEIGHTS:
        if len(w) < SRV_PER_CLUS * NUM_CLUS:
            POSSIBLE_WEIGHTS.remove(w)

    return POSSIBLE_WEIGHTS


class Agent:
    def __init__(self, env, optimizer, url="http://192.168.1.134/status/format/json"):

        # Initialize atributes
        self._state_size = env[0]  # input_shape - len(state)
        self._action_size = env[1]  # output_shape - SRV_PER_CLUS * NUM_CLUS
		self.actions = [i for i in range(self._action_size)] # all possible actions
        self._optimizer = optimizer
		self.url = url

        self.expirience_replay = deque(maxlen=2000)

        # Initialize discount and exploration rate
        self.gamma = 0.6
        self.epsilon = 0.1

        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.align_target_model()

	'''get state from the load balancer'''
	def get_state(self):
		"""get the json for the current status"""
		UPSTREAM_ZONES = ["compose_backend",
						"home_read_backend", "user_read_backend"]
		NUM_SERVERS = 3
		UPSTREAM_KEYS = ["requestCounter", "inBytes", "outBytes", "responses",
						"requestMsecCounter", "weight", "maxFails", "failTimeout", "backup", "down"]
		for _ in range(2000):  # collect 2000 samples
			status = requests.get(self.url).json()
			data = status["upstreamZones"]
			# print(status["serverZones"]["localhost"]["requestCounter"])
			for zone in UPSTREAM_ZONES:
				tmp = []
				for srv in range(NUM_SERVERS):
					for k in data[zone][srv]:
						non_2xx_3xx_responses = 0
						if k in UPSTREAM_KEYS:
							if k == "responses":
								for k1 in data[zone][srv][k]:
									if k1 in ["1xx", '4xx', '5xx']:
										non_2xx_3xx_responses += data[zone][srv][k][k1]
										tmp.append(non_2xx_3xx_responses)
									elif k == "backup":
										if data[zone][srv][k]:
											tmp.append(1)
										else:
											tmp.append(0)
									elif k == "down":
										if data[zone][srv][k]:
											tmp.append(1)
										else:
											tmp.append(1)
									else:
										tmp.append(data[zone][srv][k])
		return tmp

    def store(self, state, action, reward, next_state, terminated):
        # Store experience
        self.expirience_replay.append(
            (state, action, reward, next_state, terminated))

    def _build_compile_model(self):
        # Create the model
        # a feedforward network
        model = Sequential()
        # prepare the data for the feedforward network
        model.add(Input(env.observation_space.shape))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))

        # Configure a model for mean-squared error regression.
        model.compile(loss='mse', optimizer=self._optimizer, metrics=['mae'])
        return model

    def align_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def act(self, state):
		if np.random.rand() > self.epsilon:
    		# greedy action
        	q_values = self.q_network.predict(state.reshape(1, self._state_size))
        	return np.argmax(q_values[0])
        # Explore
		a = np.random.choice(self.actions)
        while a == q_values[0]:
			a = np.random.choice(self.actions)
		return a

    def retrain(self, batch_size):
        # pick random samples from experience memory and train the network
        minibatch = random.sample(self.expirience_replay, batch_size)

        for state, action, reward, next_state, terminated in minibatch:

            target = self.q_network.predict(state.reshape(1, self._state_size))

            if terminated:
                target[0][action] = reward
            else:
                t = self.target_network.predict(
                    next_state.reshape(1, self._state_size))
                target[0][action] = reward + self.gamma * np.amax(t)

            self.q_network.fit(state.reshape(
                1, self._state_size), target, epochs=1, verbose=0)

	def predict_target(self, state):
		predictions = self.target_network.predict(
			state.reshape(1, self._state_size))
		return np.argmax(tf.nn.softmax(predictions).numpy())

	# def dqn_training(agent, states):
		# agent.


''' Clustering part '''
NUM_CLUS = 27

ZONES = ['compose_backend', 'user_read_backend', 'home_read_backend']
SERVERS = ['192.168.1.137:8080', '192.168.1.138:8080', '192.168.1.139:8080']
METRICS = ['outBytes', 'inBytes', 'down',
           'Non 2xx/3xx responses', 'requestMsec']
CONNECTION_STATUS = ["active"]
CONTROL = ['weight']
COLS = []
ACTIONS = []

def gen_cols():
    for z in ZONES:
        for srv in SERVERS:
            for metric in METRICS + CONTROL:
                COLS.append("{}_{}_{}".format(z, srv, metric))
    for name in CONNECTION_STATUS:
        COLS.append("connections_{}".format(name))


def get_data(filename):
    """load data for analysis"""
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line))
        return data


def arrange_data(data):
    tmp = {k: [] for k in COLS}
    for entry in data:
        for z in ZONES:
            for srv in SERVERS:
                for metric in METRICS + CONTROL:
                    tmp["{}_{}_{}".format(z, srv, metric)
                        ].append(entry[z][srv][metric])
        for c in CONNECTION_STATUS:
            tmp["connections_{}".format(c)].append(
                entry["connections"][c])
    return pd.DataFrame.from_dict(tmp)


NON_STATE_COLS = []

def non_state_cols():
    for z in ZONES:
        for srv in SERVERS:
            NON_STATE_COLS.append("{}_{}_weight".format(z, srv))


def create_cluster(X):
    gmm = GaussianMixture(n_components=NUM_CLUS, reg_covar=1e-3).fit(X)
    return gmm.predict(X)


def create_actions():
    for w in POSSIBLE_WEIGHTS:
        ACTIONS.append(w * 3)


def actions_index(x):
    for i, elem in enumerate(ACTIONS):
        if (elem == x).all():
            return i

def create_rewards():
    '''we consider the variability of the last batch of rewards as a performance measure'''
	data = get_data(filename)
	gen_cols()
	non_state_cols()
	dataframe = arrange_data(data)
	df_normed = (dataframe - dataframe.mean()) / (dataframe.max() - dataframe.min())
	cols = []
	for col in df_normed.columns:
		if np.isnan(df_normed[col]).any():
			cols.append(col)
	df_normed.drop(cols, axis=1, inplace=True)
	states = create_cluster(df_normed.drop(NON_STATE_COLS, axis=1))
	# gmm = GaussianMixture(n_components=NUM_CLUS, reg_covar=1e-3).fit(df_normed.drop(NON_STATE_COLS, axis=1))
	# states = gmm.predict(df_normed.drop(NON_STATE_COLS, axis=1))
	transition_map = dataframe[NON_STATE_COLS]
	transition_map["state"] = states
	dataframe["state"] = states
	cols = []
	for col in dataframe.columns:
		if col.find('requestMsec') > -1:
			cols.append(col)

	MSECS = {}
	for s in np.unique(states):
		MSECS[s] = dataframe[cols][dataframe['state'] == s].sum(axis=1).mean() + 3 * dataframe[cols][dataframe['state'] == s].sum(axis=1).std()
	REWARDS = {}
	for i, v in enumerate(sorted(MSECS, key=MSECS.get)):
		REWARDS[v] = -i
	return MSECS, REWARDS

def get_reward(mSec, MSECS, REWARDS):
    res = [0] * len(MSECS)
	for k, v in MSECS.items():
		if v - ms < 0:
			res[k] = max(MSECS.values())
		else:
			res[k] = v - ms

    return REWARDS[np.argmin(res)]



"""
Write to nginx
"""
def change_weights(weight):
    signal.signal(signal.SIGINT, signal_handler)

    # getting the file
    config = ng.load(open("/etc/nginx/nginx.conf"))

    # getting the weights
    http = []
    http.extend(config[-1][1])

    upstreams = {}

    for item in http:
        if len(item) > 1 and isinstance(item[0], list):
            if item[0][0] == "upstream":
                tmp = []
                for i in item[1]:
                    if i[0] == "server":
                        tmp.append(i[1].split())
                upstreams[item[0][1]] = tmp

    # changing the weights
	# w = list(w)  # added
	print("Running with", w)
	time.sleep(wait_time)
	for _, v in upstreams.items():
		for i in range(len(v)):
			# v[i][1] = "weight={}".format(w[i]) # added
			v[i][1] = "weight={}".format(weight.pop())
	# formatting weights
	NEW_VALS = []
	for _, v in upstreams.items():
		for i in range(len(v)):
			NEW_VALS.append("{} {}".format(v[i][0], v[i][1]))
	# putting weights into the http list
	j = 0  # track which val from NEW_VALS to get
	for item in http:
		if len(item) > 1 and isinstance(item[0], list):
			if item[0][0] == "upstream":
				tmp = []
				for i in item[1]:
					if i[0] == "server":
						i[1] = NEW_VALS[j]
						j += 1
	# putting weights into config
	config[-1][1] = http

	# write out to the file
	out = open("/etc/nginx/nginx.conf", 'w')
	ng.dump(config, out)

	# restart the nginx process
	subprocess.call(["nginx", "-s", "reload"])





if __name__ == "__main__":
	MSEC_COUNTER = 4 # position of response time data in state
    STATES = get_state()
    POSSIBLE_WEIGHTS = gen_control_list()
	MSECS, REWARDS = create_rewards()
    env = [len(STATES[0]), len(POSSIBLE_WEIGHTS)]
    optimizer = Adam(learning_rate=0.01)
    agent = Agent(env, optimizer)

	while True:
		state = agent.get_state()
		action = POSSIBLE_WEIGHTS[agent.predict_target(state)]
		# agent.act()
		# write to nginx
		time.sleep(0.005) # sleep for 5 ms
		next_state = agent.get_state()
		reward = get_reward(state[4], MSECS, REWARDS)

		# write to nginx
		change_weights(action)

		next_state, reward, terminated = env.step(action)
		# REWARDS.append(reward)
		agent.store(state, action, reward, next_state, terminated)

		state = next_state

		batch_size = 100
		num_of_episodes = 5
		timesteps_per_episode = 20
		
		STATES = []
		REWARDS = []
		for e in range(num_of_episodes):
			# Reset the env
			# state = env.reset()

			# Initialize variables
			reward = 0
			terminated = False

			for timestep in range(timesteps_per_episode):
				# Run Action
				action = agent.act(state)

				# Take action
				# step() is a function provided by OpenAIGym to step through the environment
				# the step function provides if episode has terminated
				state = agent.get_state()
				agent.act()
				
				time.sleep(0.005) # sleep for 5 ms
				next_state = agent.get_state()
				reward = get_reward(state[4], MSECS, REWARDS)

				# write to nginx

				next_state, reward, terminated = env.step(action)
				REWARDS.append(reward)
				agent.store(state, action, reward, next_state, terminated)

				state = next_state

				STATES.append(state)  # store all steps

				if terminated:
					agent.align_target_model()
					break

				if len(agent.expirience_replay) > batch_size:
					agent.retrain(batch_size)

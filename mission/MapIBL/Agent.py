from operator import and_
from speedyibl import Agent
from collections import deque
class AgentIBL(Agent):

	# """ Agent """
	def __init__(self, outputs, default_utility = 0.1, Hash = True, delay_feedback = True):
		super(AgentIBL, self).__init__(default_utility=default_utility)
		# '''
		# :param dict config: Dictionary containing hyperparameters
		# '''
		self.outputs = outputs
		self.options = {}
		self.episode_history = []
		self.hash = Hash
		self.delay_feedback = delay_feedback
		self.last_action = 1
		self.option = 1

	# def generate_options(self,s_hash):
	# 	self.options[s_hash] = [(s_hash, a) for a in range(self.outputs)]
	
	def move(self, o, explore=True):
		# '''
		# Returns an action from the IBL agent instance.
		# :param tensor: State/Observation
		# '''
		if self.hash:
			s_hash = hash(o.tobytes())
		else:
			s_hash = o
		# if s_hash not in self.options:
			# self.generate_options(s_hash)
		# options = self.options[s_hash]
		options = [(s_hash, a) for a in range(self.outputs)]
		choice = self.choose(options)
		self.last_action = choice[1]

		self.current = s_hash

		return self.last_action



	def feedback(self, reward, nocol):

		self.respond(reward)

		#episode history
		# if self.delay_feedback and (len(self.episode_history) == 0 or self.current != self.episode_history[-1][0]) and reward !=-0.05:
		# if self.delay_feedback and (not nocol):
		if self.delay_feedback and nocol:
			self.episode_history.append((self.current,self.last_action,reward,self.t))
			

	def delayfeedback(self, reward):
		if len(self.episode_history):		
			self.equal_delay_feedback(reward, self.episode_history)
	
	
	# def getAction(self, dx, dy):
	# 	# '''
	# 	# Method that deterimines the direction 
	# 	# that the agent should take
	# 	# based upon the action selected. The
	# 	# actions are:
	# 	# 'Up':0, 
	# 	# 'Right':1, 
	# 	# 'Down':2, 
	# 	# 'Left':3, 
	# 	# 'NOOP':4
	# 	# :param action: int
	# 	# '''
	# 	# action = 4
	# 	if dx==0 and dy==-1:
	# 		action = 0
	# 	if dx==1 and dy==0:
	# 		action = 1
	# 	if dx==0 and dy==1:
	# 		action = 2
	# 	if dx==-1 and dy==0:
	# 		action = 3
	# 	if dx==0 and dy==0:
	# 		action = 4

	# 	return action
	# 	# if action == 0:
	# 	# 	return 0, -1
	# 	# elif action == 1:
	# 	# 	return 1, 0    
	# 	# elif action == 2:
	# 	# 	return 0, 1    
	# 	# elif action == 3:
	# 	# 	return -1, 0 
	# 	# elif action == 4:
	# 	# 	return 0, 0  
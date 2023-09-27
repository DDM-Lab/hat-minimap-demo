import numpy as np
import random
import cv2
import copy
import time
import collections 
# from collections import deque

class MAPWORLD(object):
	""" Cooperative multi-agent transporation problem. """
	def __init__(self,NUMBER_OF_AGENTS):
		# '''
		# :param version: Integer specifying which configuration to use
		# '''
		from mission.MapIBL.minimap import EnvConfig
		# from minimap import EnvConfig
		self.c = EnvConfig(NUMBER_OF_AGENTS)
		# Fieldnames for stats
		self.fieldnames = ['Episode',
						   'Steps',
						   'Green 1',
						   'Green 2',
				 			'Yellow',
				 			'Red',
							'Rubble',
							'Door']

		self.__dim = self.c.DIM     # Observation dimension
		self.__out = self.c.ACTIONS # Number of actions
		self.episode_count = 0      # Episode counter
		
		# Used to add noise to each cell
		self.ones = np.ones(self.c.DIM, dtype=np.float64)
		DIM = np.copy(self.c.DIM)
		DIM = np.append(DIM,3)
		self.im = np.ones(DIM,dtype = np.float64)*128

		self.X_HUMAN = [[],[]]
		self.Y_HUMAN = [[],[]]
		# self.wide_view = 100 #Observ other players
		self.wide_view = 2
		self.dir = 50
		self.teammate_view = 5

		# for y, x in self.c.OBSTACLES_YX:
		# 	self.im[y][x] = (0,0,0)
		

	@property
	def dim(self):
		return self.__dim

	@property
	def out(self):
		return self.__out

	def render(self):
		'''
		Used to render the env.
		'''
		r = 16 # Number of times the pixel is to be repeated
		try:
			# for y in range(self.c.GH):
			# 	for x in range(self.c.GW):
			# 		if self.s_t[y][x]==self.c.AGENTS[0]:
			# 			self.im[y][x] = (128, 128, 128)
			# 		elif self.s_t[y][x]==self.c.AGENTS[1]:
			# 			self.im[y][x] = (250,0,0)
			# 		elif self.s_t[y][x]==self.c.OBSTACLE:
			# 			self.im[y][x] = (0, 0, 0)
			
			# img = np.repeat(np.repeat(self.im, r, axis=0), r, axis=1).astype(np.uint8)
			img = np.repeat(np.repeat(self.s_t, r, axis=0), r, axis=1).astype(np.uint8)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			cv2.imshow('image', img)
			k = cv2.waitKey(1)
			if k == 27:         # If escape was pressed exit
				cv2.destroyAllWindows()
		except AttributeError:
			pass

	def stats(self):
		'''
		Returns stats dict
		'''
		stats = {'Episode': str(self.episode_count), 
				 'Steps': str(self.steps), 
				 'Green 1':str(self.goal_total[2]),
				 'Green 2':str(self.green_2),
				 'Yellow':str(self.goal_total[1]),
				 'Red':str(self.goal_total[0]),
				 'Rubble':str(self.rubble),
				 'Door':str(self.door)}
		return stats

	def result(self):
		stats = [self.episode_count, 
				 self.steps,
				 self.goal_total[2],
				 self.green_2,
				 self.goal_total[1],
				 self.goal_total[0],
				 self.rubble,
				 self.door]
		return stats 

	def reset(self):
		'''
		Reset everything. 
		'''
		# Set up the state array:
		# for y, x, r in self.c.GOALS_YX:
		# 	if r() == 0.75:
		# 		self.im[y][x] = (255, 255, 0)
		# 	else:
		# 		self.im[y][x] = (0,255,0)
		# 0 = obstacles, 1 = goods, 2 = agents, 3 = self
		self.s_t = np.zeros(self.c.DIM, dtype=np.float64)
		self.collections = []
		
		# self.xy_human =[]
		# self.s_goals = np.zeros(self.c.DIM, dtype=np.float64)

		# Obstacles, agents and goods are initialised:
		self.setObstacles()
		self.initGoals()
		self.initAgents()
		self.setRubbles()
		self.initHuman()
		self.setDoors()

		# Used to keep track of the reward total acheived throughout 
		# the episode:
		self.goal_total = [0,0,0]
		self.green_2 = 0
		self.rubble = 0
		self.door = 0
		# self.actions = {0:deque([], maxlen=1), 1: deque([], maxlen=1)}
		#green 5 click, red and yellow 10 click

		# Episode counter is incremented:
		self.episode_count += 1
   
		# For statistical purposes:
		# Step counter for the episode is initialised
		self.steps = 0 
		self.completed = False
		# Number of steps the goods is carried by both agents
		# self.coopTransportSteps = 0 

		# Moves taken in the same direction while carrying the goods
		# self.coordinatedTransportSteps = 0 

		self.start_time = time.time()
		self.wait_count = 0
	  
		return self.getObservations()

	def terminal(self):
		'''
		Find out if terminal conditions have been reached.
		'''
		return self.completed

	def step(self, actions):
		'''
		Change environment state based on actions.
		:param actions: list of integers
		'''
		# Agents move according to actions selected
		rewards, NoCols = self.moveAgents(actions)

		# Observations are loaded for each agent
		# observations = self.getObservations()
		r = 0 
		for i in range(self.c.NUMBER_OF_AGENTS[1]):
			y, x = self.agents_y[1][i],self.agents_x[1][i]
			goal = None 
			
			# #Ngoc added: see a red victim and then wait for the medic
			if self.c.GOAL[0] in self.s_t[max(y-self.wide_view,0):min(y+self.wide_view+1,self.c.GH),max(x-self.wide_view,0):min(x+self.wide_view+1,self.c.GW)] \
				and self.wait_count < 10:
				# print('Count: ', self.wait_count)
				self.wait_count += 1
				goal = self.c.GOAL[0]
				path, find = self.bfs_e(x,y,goal)
				red_view = 5
				if find and \
					self.c.AGENTS[0] in self.s_t[max(y-red_view,0):min(y+red_view+1,self.c.GH),max(x-red_view,0):min(x+red_view+1,self.c.GW)]:
					for ii in range(1,len(path)):
						tmp1= self.noCollision_e(path[ii][0], path[ii][1])
						if tmp1:
							if ii == len(path)-1 and self.s_t[path[ii][1]][path[ii][0]] == goal:	
								self.collections.append(path[ii])
								r += self.c.SCORES[0]
							self.moveAgent(path[ii][0], path[ii][1], 1, i)
							self.wait_count = 0
						else:
							break 
				else:
					self.start_time = time.time()
					self.wait_count = -1
			elif self.c.GOAL[2] in self.s_t[max(y-self.wide_view,0):min(y+self.wide_view+1,self.c.GH),max(x-self.wide_view,0):min(x+self.wide_view+1,self.c.GW)]:
				goal = self.c.GOAL[2]
				path, find = self.bfs_e(x,y,goal)
				if find and len(path)<5:
					for ii in range(1,len(path)):
							tmp1= self.noCollision_e(path[ii][0], path[ii][1])
							if tmp1:
								if ii == len(path)-1 and self.s_t[path[ii][1]][path[ii][0]] == goal:	
									self.collections.append(path[ii])
									self.green_2 += 1
									r += self.c.SCORES[2]
								# time.sleep(0.1)
								# print('Near green....', (path[ii][0], path[ii][1]))
								self.moveAgent(path[ii][0], path[ii][1], 1, i)
							else:
								break
			elif self.c.DOOR in self.s_t[max(y-self.wide_view,0):min(y+self.wide_view+1,self.c.GH),max(x-self.wide_view,0):min(x+self.wide_view+1,self.c.GW)]:
				goal = self.c.DOOR
				path, find = self.bfs_e(x,y,goal)
				if find and len(path)<5:
					for ii in range(1,len(path)):
							tmp1= self.noCollision_e(path[ii][0], path[ii][1])
							if tmp1:
								if ii == len(path)-1 and self.s_t[path[ii][1]][path[ii][0]] == goal:	
									self.collections.append(path[ii])
									self.door += 1
								
								self.moveAgent(path[ii][0], path[ii][1], 1, i)
							else:
								break
			elif self.c.RUBBLE in self.s_t[max(y-self.wide_view,0):min(y+self.wide_view+1,self.c.GH),max(x-self.wide_view,0):min(x+self.wide_view+1,self.c.GW)]:
				goal = self.c.RUBBLE
				path, find = self.bfs_e(x,y,goal)
				if find and len(path)<5:
					for ii in range(1,len(path)):
							tmp1= self.noCollision_e(path[ii][0], path[ii][1])
							if tmp1:
								if ii == len(path)-1 and self.s_t[path[ii][1]][path[ii][0]] == goal:	
									self.collections.append(path[ii])
									self.rubble += 1
								
								self.moveAgent(path[ii][0], path[ii][1], 1, i)
							else:
								break 
			
			elif self.c.AGENTS[0] in  self.s_t[max(y-self.teammate_view,0):min(y+self.teammate_view+1,self.c.GH),max(x-self.teammate_view,0):min(x+self.teammate_view+1,self.c.GW)]:
				goal = self.c.AGENTS[0]
				path, find = self.bfs_e(x,y,goal)
				if find and len(path)<5*2:
					for ii in range(0,len(path)-1):
						tmp1= self.noCollision_e(path[ii][0], path[ii][1])
						if tmp1:
							if ii == len(path)-1 and self.s_t[path[ii][1]][path[ii][0]] == goal:	
								self.colections.append(path[ii])
								break
							# print('Near medic....', (path[ii][0], path[ii][1]))
							self.moveAgent(path[ii][0], path[ii][1], 1, i)
						# else:
						# 	break 
			
			

			# #guide agents
			if goal is None:
				if self.c.GOAL[0] in self.s_t[max(y-self.dir,0):min(y+self.dir,self.c.GH),max(x-self.dir,0):min(x+self.dir,self.c.GW)] and self.wait_count!=-1 \
					and time.time()-self.start_time < 80:
					goal = self.c.GOAL[0]
				elif self.c.GOAL[2] in self.s_t[max(y-self.dir,0):min(y+self.dir,self.c.GH),max(x-self.dir,0):min(x+self.dir,self.c.GW)]:
					goal = self.c.GOAL[2]
				elif self.c.DOOR in self.s_t[max(y-self.dir,0):min(y+self.dir,self.c.GH),max(x-self.dir,0):min(x+self.dir,self.c.GW)]:
					goal = self.c.DOOR
				elif self.c.RUBBLE in self.s_t[max(y-self.dir,0):min(y+self.dir,self.c.GH),max(x-self.dir,0):min(x+self.dir,self.c.GW)]:
					goal = self.c.RUBBLE
				# elif self.c.AGENTS[0] in self.s_t[max(y-self.dir,0):min(y+self.dir,self.c.GH),max(x-self.dir,0):min(x+self.dir,self.c.GW)]:
				# 	goal = self.c.AGENTS[0]
				
				
				if goal is not None:
					path, find = self.bfs_e(x,y,goal)
					if find:
						for ii in range(4):
							tmp1= self.noCollision_e(path[ii][0], path[ii][1])
							if tmp1:
								self.moveAgent(path[ii][0], path[ii][1], 1, i)
							else:
								break 

    	
		for i in range(self.c.NUMBER_OF_AGENTS[0]):
			y, x = self.agents_y[0][i],self.agents_x[0][i]
			goal = None 
			for j in range(3):
				if self.c.GOAL[j] in self.s_t[max(y-self.wide_view,0):min(y+self.wide_view+1,self.c.GH),max(x-self.wide_view,0):min(x+self.wide_view+1,self.c.GW)]:
					goal = self.c.GOAL[j]
					break

			if goal is not None:
				path, find = self.bfs_m(x,y,goal)
				if find:
					if (j!=0) or self.Near(path[-1][0], path[-1][1]):
					# if (j==0) and self.Near(path[-1][0], path[-1][1]):
						self.goal_total[j] += 1
						r += self.c.SCORES[j]
						self.moveAgent(path[-1][0], path[-1][1], 0, i)
						self.collections.append(path[-1])
			
			# #guide Medic agents
			if goal is None:
				# for j in range(1,3):
				for j in range(3):
					if self.c.GOAL[j] in self.s_t[max(y-self.dir,0):min(y+self.dir,self.c.GH),max(x-self.dir,0):min(x+self.dir,self.c.GW)]:
						goal = self.c.GOAL[j]
						print(f'Medic goal {goal} in self.dir radius...')
						break

				if goal is not None:
					path, find = self.bfs_m(x,y,goal)
					if find:
						for ii in range(1,3):
							tmp1 = self.noCollision(path[ii][0], path[ii][1])
							if tmp1:
								self.moveAgent(path[ii][0], path[ii][1], 0, i)
							else:
								break 
			
		observations = self.getObservations()
		self.steps += 1 
		
		return observations, rewards+r, self.terminal(), NoCols

	def update_human(self, location, role, ind):
		'''
		Change environment state based on actions.
		:param actions: list of integers
		'''
		x1, y1 = location[0][0], location[0][1]
		x2, y2 = location[1][0], location[1][1]
		reward = self.humanPickup(y2,x2,role)

		self.s_t[y1][x1] -= self.c.AGENTS[role]
		
		self.s_t[y2][x2] = self.c.AGENTS[role]

		self.x_human[role][ind] = x2
		self.y_human[role][ind] = y2
		return reward
	
	def set_human(self, x, y, role):
		'''
		Change environment state based on actions.
		:param actions: list of integers
		'''
		self.X_HUMAN[role].append(x)
		self.Y_HUMAN[role].append(y)
		# self.s_t[y][x] = self.c.AGENTS[role]

	def initGoals(self):
		# '''
		# Goods position and carrier ids are initialised
		# '''
		for (y, x, r) in self.c.GOALS_YX:
			for i in range(3):
				if r() == self.c.SCORES[i]:
					self.s_t[y][x] = self.c.GOAL[i]
					

	def initAgents(self):
		# '''
		# Method for initialising the required number of agents and 
		# positionsing them on designated positions within the grid
		# '''

		self.agents_x = copy.deepcopy(self.c.AGENTS_X)
		self.agents_y = copy.deepcopy(self.c.AGENTS_Y)		
   
		# Agents are activated within the agent channel (2) of the gridworld
		# matrix.
		for t in range(2):
			for i in range(self.c.NUMBER_OF_AGENTS[t]):
				self.s_t[self.agents_y[t][i]][self.agents_x[t][i]] += self.c.AGENTS[t]
	def initHuman(self):
		# '''
		# Method for initialising the required number of agents and 
		# positionsing them on designated positions within the grid
		# '''

		self.x_human = copy.deepcopy(self.X_HUMAN)
		self.y_human = copy.deepcopy(self.Y_HUMAN)

		for t in range(2):
			for x, y in zip(self.x_human[t], self.y_human[t]):
				self.s_t[y][x] += self.c.AGENTS[t]

	def setObstacles(self):
		'''
		Method used to initiate the obstacles within the environment 
		'''
		for y, x in self.c.OBSTACLES_YX:
			self.s_t[y][x] = self.c.OBSTACLE
	def setRubbles(self):
		'''
		Method used to initiate rubble within the environment 
		'''
		for y, x in self.c.RUBBLES_YX:
			self.s_t[y][x] = self.c.RUBBLE
	
	def setDoors(self):
		'''
		Method used to initiate door within the environment 
		'''
		for y, x in self.c.DOORS_YX:
			self.s_t[y][x] = self.c.DOOR

	def goalsPickup(self, y, x, t, i, a):
		'''
		Method to check one of the goods 
		has been picked up. 
		t: index of role (0: medic; 1:engineer)
		i: number of agents per role
		a: action
		'''
		
		r = 0
		if t==0:
			if a == 4:
				for j in range(3):
					if self.c.GOAL[j] in self.s_t[y,max(x-1,0):min(x+2,self.c.GW)]:
						dxg = np.where(self.s_t[y,max(x-1,0):min(x+2,self.c.GW)]==self.c.GOAL[j])[0][0] - 1
						if (j!=0) or self.Near(x+dxg,y):
							self.goal_total[j] += 1
							r = self.c.SCORES[j]
							self.moveAgent(x+dxg,y, 0, i)
							self.collections.append((x+dxg,y))
							break
					elif self.c.GOAL[j] in self.s_t[max(y-1,0):min(y+2,self.c.GH),x]:
						dyg = np.where(self.s_t[max(y-1,0):min(y+2,self.c.GH),x]==self.c.GOAL[j])[0][0] - 1
						if (j!=0) or self.Near(x,y+dyg):
							self.goal_total[j] += 1
							r = self.c.SCORES[j]
							self.moveAgent(x,y+dyg, 0, i)
							self.collections.append((x,y+dyg))
							break
		elif t==1:
			if a == 4:
				if self.c.GOAL[2] in self.s_t[y,max(x-1,0):min(x+2,self.c.GW)]:
					dxg = np.where(self.s_t[y,max(x-1,0):min(x+2,self.c.GW)]==self.c.GOAL[2])[0][0] - 1
					self.green_2 += 1
					r = self.c.SCORES[2]
					self.moveAgent(x+dxg,y,1,i)
					self.collections.append((x+dxg,y))
				elif self.c.GOAL[2] in self.s_t[max(y-1,0):min(y+2,self.c.GH),x]:
					dyg = np.where(self.s_t[max(y-1,0):min(y+2,self.c.GH),x]==self.c.GOAL[2])[0][0] - 1
					
					self.green_2 += 1
					r = self.c.SCORES[2]
					self.moveAgent(x,y+dyg,1, i)
					self.collections.append((x,y+dyg))
				elif self.c.RUBBLE in self.s_t[y,max(x-1,0):min(x+2,self.c.GW)]:
					dxg = np.where(self.s_t[y,max(x-1,0):min(x+2,self.c.GW)]==self.c.RUBBLE)[0][0] - 1
					self.moveAgent(x+dxg,y, 1, i)
					self.collections.append((x+dxg,y))
				elif self.c.RUBBLE in self.s_t[max(y-1,0):min(y+2,self.c.GH),x]:
					dyg = np.where(self.s_t[max(y-1,0):min(y+2,self.c.GH),x]==self.c.RUBBLE)[0][0] - 1
					self.moveAgent(x,y+dyg, 1, i)
					self.collections.append((x,y+dyg))

				elif self.c.DOOR in self.s_t[y,max(x-1,0):min(x+2,self.c.GW)]:
					dxg = np.where(self.s_t[y,max(x-1,0):min(x+2,self.c.GW)]==self.c.DOOR)[0][0] - 1
					self.door += 1
					self.moveAgent(x+dxg,y, 1, i)
					self.collections.append((x+dxg,y))
				elif self.c.DOOR in self.s_t[max(y-1,0):min(y+2,self.c.GH),x]:
					dyg = np.where(self.s_t[max(y-1,0):min(y+2,self.c.GH),x]==self.c.DOOR)[0][0] - 1
					self.door += 1
					self.moveAgent(x,y+dyg, 1, i)
					self.collections.append((x,y+dyg))
		
		if sum(self.goal_total)+ self.green_2 == self.c.number_goals:
			self.completed = True                
		return r

	def humanPickup(self, y, x, i):
		# '''
		# Method to check one of the goods 
		# has been deliverd to the dropzone
		# '''
		# for (dropX, dropY, r) in self.c.GOALS_YX:
		r = 0
		if i==0:
			for j in range(3):
				if self.s_t[y][x] == self.c.GOAL[j]:
					self.goal_total[j] += 1
					r = self.c.SCORES[j]
					self.collections.append((x,y))
					break
		else:
			if self.s_t[y][x] == self.c.GOAL[2]:
				self.green_2 += 1
				r = self.c.SCORES[2]
				self.collections.append((x,y))
		
		if sum(self.goal_total)+ self.green_2 == self.c.number_goals:
			self.completed = True                
			
		return r

	

	# def getObservations(self):
	# 	# '''
	# 	# Returns centered observation for each agent
	# 	# '''
	# 	return (tuple(self.agents_x[0]+self.x_human[0]),tuple(self.agents_y[0]+self.y_human[0]), tuple(self.agents_x[1]+self.x_human[1]),tuple(self.agents_y[1]+self.y_human[1]), tuple(self.collections)) #[np.copy(self.s_t), np.copy(self.s_t)]

	def getObservations(self):
		# '''
		# Returns centered observation for each agent
		# '''
		observations = [[], []]
		for t in range(2):
			for i in range(self.c.NUMBER_OF_AGENTS[t]):
				# observations[t][i] = []
				tmp = [[], []]
				for tt in range(2):
					for x, y in zip(self.agents_x[tt] + self.x_human[tt], self.agents_y[tt] + self.y_human[tt]):
						if abs(x-self.agents_x[t][i]) > 0 and abs(x-self.agents_x[t][i])<= self.wide_view and abs(y - self.agents_y[t][i]) > 0 and abs(y - self.agents_y[t][i]) <= self.wide_view:
							tmp[tt].append((x,y))
				observations[t].append(((self.agents_x[t][i],self.agents_y[t][i]),tuple(tmp[0]),tuple(tmp[1]), tuple(self.collections)))
		return observations 
		# return self.s_t 
		# return (tuple(self.agents_x[0]+self.x_human[0]),tuple(self.agents_y[0]+self.y_human[0]), tuple(self.agents_x[1]+self.x_human[1]),tuple(self.agents_y[1]+self.y_human[1]), tuple(self.collections)) #[np.copy(self.s_t), np.copy(self.s_t)]

	def getDelta(self, action):
		# '''
		# Method that deterimines the direction 
		# that the agent should take
		# based upon the action selected. The
		# actions are:
		# 'Up':0, 
		# 'Right':1, 
		# 'Down':2, 
		# 'Left':3, 
		# 'NOOP':4
		# :param action: int
		# '''
		if action == 0:
			return 0, -1
		elif action == 1:
			return 1, 0    
		elif action == 2:
			return 0, 1    
		elif action == 3:
			return -1, 0 
		elif action == 4: #interact
			return 0, 0   

	def moveAgents(self, actions):
	#    '''
	#    Move agents according to actions.
	#    :param actions: List of integers providing actions for each agent
	#    '''

		# dx = []
		# dy = []


		r_tmp =0
		# r = [[],[]]
		tmp = [[],[]]
		# score_m = 0
		for t in range(2):
			# if t == 1:
				# for r1 in r[0]:
					# if r1==self.c.SCORES[0] or r1==self.c.SCORES[1]:
						# score_m = score_m + r1
			for i in range(self.c.NUMBER_OF_AGENTS[t]):
				if len(actions[t]) == 0:
					actions[t].append(4) #interact
				r_tmp += self.goalsPickup(self.agents_y[t][i],self.agents_x[t][i],t,i,actions[t][i])
				tmp[t].append(False)
				if actions[t][i] != 4:
					dx , dy = self.getDelta(actions[t][i])
					x = self.agents_x[t][i] + dx
					y = self.agents_y[t][i] + dy
					tmp1, tmp2 = self.noCollision(x, y)
					tmp[t][i] = tmp2 
					if tmp1:
						# print('move')
						self.moveAgent(x, y, t, i)
		
		return r_tmp, tmp 
	
	def moveAgent(self, x, y, t, i):
		# '''
		# Moves agent to target x and y
		# :param targetx: Int, target x coordinate
		# :param targety: Int, target y coordinate
		# '''
		# self.old_agents_x = np.copy(self.agents_x)
		# self.old_agents_y = np.copy(self.agents_y)
		# for i in range(self.c.NUMBER_OF_AGENTS):
			
			# self.s_t[self.agents_y[i]][self.agents_x[i]] -= self.c.AGENTS[i]
			# self.agents_x[i] = targetx[i]
			# self.agents_y[i] = targety[i]
			# self.s_t[self.agents_y[i]][self.agents_x[i]] = self.c.AGENTS[i]
		self.s_t[self.agents_y[t][i]][self.agents_x[t][i]] -= self.c.AGENTS[t]
		self.agents_x[t][i] = x
		self.agents_y[t][i] = y
		# print('move agent',i,'x-y-old',self.old_agents_x[i],self.old_agents_y[i],'x-y-new',self.agents_x[i],self.agents_y[i])
		# time.sleep(0.1)
		self.s_t[self.agents_y[t][i]][self.agents_x[t][i]] = self.c.AGENTS[t]

	def noCollision(self, x, y):
		# '''
		# Checks if x, y coordinate is currently empty 
		# :param x: Int, x coordinate
		# :param y: Int, y coordinate
		# '''
		
		tmp1 = self.noCollisionHard(x, y)

		if tmp1 and self.s_t[y][x] ==0:
			tmp = True 
		else:
			tmp = False	

		return tmp, tmp1
	
	def noCollisionHard(self, x, y):
		if x < 0 or x >= self.c.GW or y < 0 or y >= self.c.GH\
			or self.s_t[y][x]==self.c.OBSTACLE\
			or self.s_t[y][x]==self.c.RUBBLE\
			or self.s_t[y][x]==self.c.DOOR\
			or self.s_t[y][x]==self.c.AGENTS[0] or self.s_t[y][x]==self.c.AGENTS[1]:
			return False
		else:
			return True
	def noCollision_e(self, x, y):
		if x < 0 or x >= self.c.GW or y < 0 or y >= self.c.GH\
			or self.s_t[y][x]==self.c.OBSTACLE\
			or self.s_t[y][x] == self.c.GOAL[0]\
			or self.s_t[y][x] == self.c.GOAL[1]\
			or self.s_t[y][x]==self.c.AGENTS[0]:
			return False
		else:
			return True
	
	def Near(self, x,y):
		tmp = False
		if self.c.AGENTS[1] in self.s_t[y,max(x-1,0):min(x+2,self.c.GW)]:
			tmp = True 
		elif self.c.AGENTS[1] in self.s_t[max(y-1,0):min(y+2,self.c.GH),x]:
			tmp = True 						
		return tmp 
	
	def near_medic(self, x, y):
		flag = False
		if self.c.AGENTS[0] in self.s_t[max(y-self.dir,0):min(y+self.dir,self.c.GH),max(x-self.dir,0):min(x+self.dir,self.c.GW)]:
			flag = True 
		return flag

	def bfs_m(self, xt, yt, goal):
		queue = collections.deque([[(xt,yt)]])
		seen = set([(xt,yt)])
		find = False
		while queue:
			path = queue.popleft()
			x, y = path[-1]
			if self.s_t[y][x] == goal:
				# return path
				find = True
				break
			for x2, y2 in ((x+1,y),(x-1,y), (x,y+1), (x,y-1)):
				if max(xt-self.wide_view,0) <= x2 <= min(x+self.wide_view+1,self.c.GW-1) and max(y-self.wide_view,0) <= y2 <= min(y+self.wide_view+1,self.c.GH-1)\
					and self.s_t[y2][x2] != self.c.OBSTACLE and (x2,y2) not in seen\
					and self.s_t[y2][x2] != self.c.DOOR\
					and self.s_t[y2][x2] != self.c.RUBBLE\
					and self.s_t[y2][x2] != self.c.AGENTS[1]:
					queue.append(path+[(x2,y2)])
					seen.add((x2,y2))
		return path, find 
	# max(y-self.wide_view,0):min(y+self.wide_view,self.c.GH),max(x-self.wide_view,0):min(x+self.wide_view,self.c.GW)
	
	def bfs_e(self, xt, yt, goal):
		queue = collections.deque([[(xt,yt)]])
		seen = set([(xt,yt)])
		find = False
		while queue:
			path = queue.popleft()
			x, y = path[-1]
			if self.s_t[y][x] == goal:
				# return path
				find = True
				break
			for x2, y2 in ((x+1,y),(x-1,y), (x,y+1), (x,y-1)):
				if max(xt-self.wide_view,0) <= x2 <= min(x+self.wide_view+1,self.c.GW-1)\
					and max(y-self.wide_view,0) <= y2 <= min(y+self.wide_view+1,self.c.GH-1)\
					and self.s_t[y2][x2] != self.c.OBSTACLE\
					and (x2,y2) not in seen\
					and self.s_t[y2][x2] != self.c.GOAL[1]:
					# and self.s_t[y2][x2] != self.c.AGENTS[0]\
					# and self.s_t[y2][x2] != self.c.DOOR\
					# and self.s_t[y2][x2] != self.c.RUBBLE 
					# and self.s_t[y2][x2] != self.c.GOAL[0]\:
					
					queue.append(path+[(x2,y2)])
					seen.add((x2,y2))
		return path, find 
import pandas as pd
class EnvConfig:

    ''' Env. Parameters '''
    def __init__(self,NUMBER_OF_AGENTS):
        """ Gridworld Dimensions """
        df = pd.read_csv('mission/MapIBL/map_new.csv')
        self.GRID_HEIGHT = df['z'].max() + 1
        self.GRID_WIDTH = df['x'].max() + 1
        # self.GRID_HEIGHT = 16
        # self.GRID_WIDTH = 16
        # self.MID = self.GRID_WIDTH // 2
        self.GH = self.GRID_HEIGHT
        self.GW = self.GRID_WIDTH
        self.DIM = [self.GH, self.GW]
        # self.HMP = int(self.GW/2) # HMP = Horizontal Mid Point
        # self.VMP = int(self.GH/2) # VMP = Vertical Mid Point
        self.ACTIONS = 5 # No-op, move up, down, left, righ

        """ Wind (slippery surface) """
        self.WIND = 0.0
       
        """ Agents """
        # self.number_m=number_m
        # self.number_e = number_e
        self.NUMBER_OF_AGENTS = NUMBER_OF_AGENTS
        self.AGENTS_X = []
        self.AGENTS_Y = []
        for t in range(2):

            self.AGENTS_X.append([12 for i in range(self.NUMBER_OF_AGENTS[t])])
            self.AGENTS_Y.append([4+t for i in range(self.NUMBER_OF_AGENTS[t])])
            # self.AGENTS = [240+i*10 for i in range(self.NUMBER_OF_AGENTS[t])]

      
        """ Goals """
        self.GOALS_YX = [] # [(2,0,lambda:1.0)] # X, Y, Reward
 
        """ Colors """
        self.AGENTS = [240.0, 180.0] # Colors [Agent1, Agent2]
        
        self.GOAL = [100.0, 170.0, 250.0]
        self.OBSTACLE = 20.0
        self.number_goals = 0
        self.RUBBLE = 60.0
        self.DOOR = 230.0

        """ Obstacles """
        self.OBSTACLES_YX = []
        self.SCORES = [0.6, 0.3, 0.1]
        self.RUBBLES_YX = []

        """ Doors """
        self.DOORS_YX = []

        self.n = df.shape[0]
        for i in range(self.n):
            # print(i)
            info_row = df.iloc[i]
            # print(i)
            if info_row['key'] in ['wall']:
                self.OBSTACLES_YX.append((info_row['z'], info_row['x']))
            elif info_row['key'] == 'rubble':
                self.RUBBLES_YX.append((info_row['z'], info_row['x']))
            elif info_row['key'] == 'door':
                self.DOORS_YX.append((info_row['z'], info_row['x']))
            elif info_row['key'] == 'red':
                self.GOALS_YX.append((info_row['z'], info_row['x'],lambda:self.SCORES[0]))
                self.number_goals += 1
            elif info_row['key'] == 'yellow':
                self.GOALS_YX.append((info_row['z'], info_row['x'],lambda:self.SCORES[1]))
                self.number_goals += 1
            elif info_row['key'] == 'green':
                self.GOALS_YX.append((info_row['z'], info_row['x'],lambda:self.SCORES[2]))
                self.number_goals += 1


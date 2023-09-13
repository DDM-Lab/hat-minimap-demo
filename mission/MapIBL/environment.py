import re

class Environment(object):

    """ Environment """
    def __init__(self,NUMBER_OF_AGENTS = [1,1]):

        from mission.MapIBL.minimapworld import MAPWORLD
        self.__env = MAPWORLD(NUMBER_OF_AGENTS)
        
        self.__fieldnames = self.__env.fieldnames
        self.__dim = self.env.dim
        self.__out = self.env.out

    def getHW(self):
        return self.__env.getHW()

    @property
    def upper_bound(self):
        return self.__upper_bound

    @property
    def lower_bound(self):
        return self.__lower_bound

    @property
    def dim(self):
        return self.__dim

    @property
    def out(self):
        return self.__out

    @property
    def env(self):
        return self.__env

    @property
    def fieldnames(self):
        return self.__fieldnames


    def reset(self):
        '''
        Reset env to original state
        '''
        return self.__env.reset()

    def render(self):
        '''
        Render the environment
        '''
        self.__env.render()

    def step(self, a):
        '''
        :param float: action
        '''
        return self.__env.step(a)
    
    def set_human(self, x, y, role):
        '''
        :param float: action
        '''
        return self.__env.set_human(x, y, role)
    
    def update_human(self, location,role, ind):
        '''
        :param float: action
        '''
        return self.__env.update_human(location,role,ind)

    def stats(self):
        '''
        :return: stats from env
        '''
        return self.__env.stats()
    
    def result(self):
        '''
        :return: stats from env
        '''
        return self.__env.result()

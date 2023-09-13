episodes = 8
steps = 2500
number_agents = 2

data = []

from environment import Environment
env = Environment()

from Agent import AgentIBL

from copy import deepcopy

from stats import mkRunDir
import csv
import random

import argparse
# import sys
# sys.argv=['']
# del sys
flags = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="lightIBL")

flags.add_argument('--environment',type=str,default='MINIMAP',help='Environment.')
flags.add_argument('--drl',type=str,default='IBL',help='Type of Agent')
# flags.add_argument('--madrl',type=str,default='hysteretic',help='MA-IBL extension. Options: None, leniency, hysteretic')
# flags.add_argument('--madrl',type=str,default='leniency',help='MA-IBL extension. Options: None, leniency, hysteretic')
flags.add_argument('--madrl',type=str,default='handed',help='MA-IBL extension. Options: handed, nohanded')
# flags.add_argument('--madrl',type=str,default='original',help='MA-IBL extension. Options: None, leniency, hysteretic')
flags.add_argument('--agents',type=int,default=2,help='Number of agents.')
flags.add_argument('--episodes',type=int,default=1000,help='Number of episodes.')
flags.add_argument('--steps',type=int,default=5000,help='Number of steps.')
flags.add_argument('--default_utility',type=float,default=0.1,help='Number of steps.') #test for V8
FLAGS = flags.parse_args()

# statscsv, folder = mkRunDir(env, runid)
statscsv, folder = mkRunDir(env)

agents = []
for i in range(number_agents): 
    agents.append(AgentIBL(env.out,default_utility=0.1,Hash = True)) # Init agent instances

for i in range(episodes):
    
# Run episode
    observations = env.reset() # Get first observations
    episode_reward = 0
    NoCols = [True,True]

    for j in range(steps):
        # env.render()
        #######################################
        actions = []
        if FLAGS.madrl == 'handed':
            for agent, o, nocol in zip(agents,observations,NoCols):
                # if nocol == True:
                if j%3==0 or nocol==False:
                    actions.append(agent.move(o))
                else:
                    actions.append(agent.last_action)

                    if agent.hash:
                        s_hash = hash(o.tobytes())
                    else:
                        s_hash = o

                    agent.t += 1
                    agent.current = s_hash
        else:
            for agent, o in zip(agents,observations):
                actions.append(agent.move(o))
        # print('actions',actions)
        observations, rewards, t, NoCols = env.step(actions)

        for agent, r, nocol in zip(agents, rewards, NoCols):
            if nocol == False:
                
                r = -0.05
            elif r==0:
                r = -0.01
            agent.feedback(r)  
            
            if r>0:
                agent.delayfeedback(r) 
                # print(agent.episode_history)
                agent.episode_history = []

        if j == steps-1:
            t = True

        episode_reward += rewards[0]
        if t: 
            for agent in agents:
                agent.episode_history = []
            break # If t then terminal state has been reached
        data.append([i, j, episode_reward])
    with open(statscsv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=env.fieldnames)
        writer.writerow(env.stats())
    print(env.stats())
# for agent in agents:
#     agent.instances()
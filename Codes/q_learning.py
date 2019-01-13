import numpy as np
import random 
import matplotlib.pyplot as plt
#set up alpha epsilon
alpha= 0.01
epsilon=0.1
num_states= 8
#for keeping track of the largest q value of the source state
Source_state=[]
#set the action table, possible action for each state
action_table= np.array([[1,2,3],[4,5],[5],[5,6],[7],[7],[7]])
#save the results of different parameters
def save_results(data, filename): 
	  with open(filename, "a") as data_file:
	  	data_file.write("%s\n" %data)
	  data_file.close()
#Set the reward table 
reward_table= np.matrix([[-np.inf,-1,-3,-5,-np.inf, -np.inf, -np.inf,-np.inf],
	[-np.inf,-np.inf,-np.inf,-np.inf,-7, -9, -np.inf,-np.inf],
	[-np.inf,-np.inf,-np.inf,-np.inf,-np.inf, -3, -np.inf,-np.inf],
	[-np.inf,-np.inf,-np.inf,-np.inf,-np.inf, -7, -2,-np.inf],
	[-np.inf,-np.inf,-np.inf,-np.inf,-np.inf, -np.inf, -np.inf,-5],
	[-np.inf,-np.inf,-np.inf,-np.inf,-np.inf, -np.inf, -np.inf,-4],
	[-np.inf,-np.inf,-np.inf,-np.inf,-np.inf, -np.inf, -np.inf,-2],
	[-np.inf,-np.inf,-np.inf,-np.inf,-np.inf, -np.inf, -np.inf,-np.inf]])
#corresponding table
Rewards = {0:'S',1:'A',2:'B',3:'C',4:'D',5:'E',6:'F',7:'T'}
#policy
policy= np.zeros(7)
#initialize q table
q_table=np.zeros((8,8))
num_episodes=10000
yes=0
#start iterate over episodes
for i in range(num_episodes):
	print('episode',i)
	#keep track of the trajectory for each episode
	trajectory = []
	#the initial state is random
	init_state= random.randint(0,6)
	state= np.asarray([init_state])
	
	
	trajectory.append(init_state)
	
	is_terminal = False
	#run until reach the terminal state
	while is_terminal== False:
		probability= random.uniform(0,1)
		#epsilon greedy method
		if (probability>epsilon):
			action = random.choice(action_table[state[0]])
		else:
			#if all q values are zero, select a random one
			if ((np.amin(q_table[state[0]]))==0):
				action=random.choice(action_table[state[0]])
			else:
			# greedy action, choose the one with the largest q value
				print('yes',q_table[state[0]])
				yes+=1
				tmp=np.array([q_table[state[0]][x] for x in action_table[state[0]] if x!=0])
				max2=np.max(tmp[np.nonzero(tmp)])
				print('tmp',tmp)
				
				action= list(q_table[state[0]]).index(max2)
		#update the q_table for each step, apply the formula
		if (np.amin(q_table[action]))==0:
			q_table[state[0],action]= q_table[state[0],action]+alpha*(reward_table[state[0],action]-q_table[state[0],action])
		
		else:

			q_table[state[0],action]= q_table[state[0],action]+alpha*(reward_table[state[0],action]+np.max(q_table[action][np.nonzero(q_table[action])])-
												q_table[state[0],action])
		#state is the previous action
		state[0]=action
		trajectory.append(state[0])
		#when we reach the terminal state
		if state[0]== 7:
			print('trajectory',trajectory)
			is_terminal= True
			print('q_table',q_table)
			#append the largest q value for the source node
			if (np.min(q_table[0])==0):
				Source_state.append(np.max(q_table[0]))
			else:
				Source_state.append(np.max(q_table[0][np.nonzero(q_table[0])]))	
			
			


print('final_q_table',q_table)
# get the best action for each state
for i in range(7):
	policy[i]=list(q_table[i]).index(np.max(q_table[i][np.nonzero(q_table[i])]))
print('final policy',policy)
#for i in range(num_episodes):
#	save_results(Source_state[i],"q_learning_alpha_0.01_epsilon_0.1.dat")

# draw the graph
plt.figure(1)
plt.plot(Source_state,label="max(Q[0,a])")
plt.xlabel("Episodes")
plt.ylabel("largest action value for Source_state")
plt.legend()
plt.show()

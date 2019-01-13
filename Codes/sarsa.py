import numpy as np
import random 
import matplotlib.pyplot as plt
#set the parameters
alpha= 0.01
epsilon=0.1
num_states= 8
#list for the largest q value of the start node for the current episode 
Source_state=[]
#save result of different parameters 
def save_results(data, filename): 
	  with open(filename, "a") as data_file:
	  	data_file.write("%s\n" %data)
	  data_file.close()
#possible action table
action_table= np.array([[1,2,3],[4,5],[5],[5,6],[7],[7],[7]])

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
#optimal policy
policy= np.zeros(7)
#initialize q table
q_table=np.zeros((8,8))
#number of episodes
num_episodes=10000
yes=0
for i in range(num_episodes):
	print('episode',i)
	#trajectory to keep track of the states passed for each episode
	trajectory = []
	init_state= random.randint(0,6)
	state= np.asarray([init_state])
	#epsilon greedy for generating action
	probability= random.uniform(0,1)
	trajectory.append(init_state)
	print("init_state",state[0])
	#epsilon greedy for generating action
	if (probability>epsilon):
			action = random.choice(action_table[state[0]])
	else:
			
		if ((np.amin(q_table[state[0]]))==0):
			action=random.choice(action_table[state[0]])
		else:
			print('yes',q_table[state[0]])
			tmp=np.array([q_table[state[0]][x] for x in action_table[state[0]] if x!=0])
			max3=np.max(tmp[np.nonzero(tmp)])
			print('tmp',tmp,'max3',max3)
				
			action= list(q_table[state[0]]).index(max3)
	#print('first action',action)
	

	is_terminal = False
	while is_terminal== False:
		#next state is the current action
		next_state= action
		print("next_state",next_state)
		#when reaching the terminal state
		if next_state==7:
			print("reach terminal state")
			trajectory.append(next_state)
			#update the q table
			q_table[state[0],action]= q_table[state[0],action]+alpha*(reward_table[state[0],action]-
												q_table[state[0],action])
			
			print("trajectory",trajectory)
			print('q_table',q_table)
			#get the max q value for the start node
			if (np.amin(q_table[0])==0):
				Source_state.append(np.max(q_table[0]))

			else:
				Source_state.append(np.max(q_table[0][np.nonzero(q_table[0])]))	
			is_terminal=True
			break
		prob= random.uniform(0,1)
		

		#epsilon greedy method
		if (prob<epsilon):
			next_action = random.choice(action_table[next_state])
		else:
			
			if ((np.amin(q_table[next_state]))==0):
				next_action=random.choice(action_table[next_state])
			else:
				print('yes',q_table[next_state])
				tmp1=np.array([q_table[next_state][x]for x in action_table[next_state] if x!=0])
				max2=np.max(tmp1[np.nonzero(tmp1)])
				print('tmp1',tmp1,'max2',max2)
				
				next_action= list(q_table[next_state]).index(max2)

		#update the q table if not in the terminal state
		print("state[0]",state[0],"action",action,"next_state",next_state,"next_action",next_action)
		q_table[state[0],action]= q_table[state[0],action]+alpha*(reward_table[state[0],action]+q_table[next_state,next_action]-
												q_table[state[0],action])
		# next state becomes the current state,next action becomes the current action
		state[0]=next_state
		action= next_action
		trajectory.append(next_state)
		
		
			
#save the results for different parameters		
#for i in range(num_episodes):
#	save_results(Source_state[i],"sarsa_alpha_0.01_epsilon_0.1.dat")

print("yes",yes)
print('final_q_table',q_table)
for i in range(7):
	policy[i]=list(q_table[i]).index(np.max(q_table[i][np.nonzero(q_table[i])]))
print('final policy',policy)
#draw the figure
plt.figure(1)
plt.plot(Source_state,label="max(Q[0,a])")
plt.xlabel("Episodes")
plt.ylabel("largest action value for Source_state")
plt.legend()
plt.show()

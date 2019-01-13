import numpy as np
import random 
import matplotlib.pyplot as plt
#set the parameters
alpha = 0.1
epsilon = 0.01
num_states= 8
#set the possible action table
action_table= np.array([[1,2,3],[4,5],[5],[5,6],[7],[7],[7]])
#function for saving the result of largest q value for the source node
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

#list for the largest q value for each episode
Source_state=[]
#numbers corresponding to the states
Rewards = {0:'S',1:'A',2:'B',3:'C',4:'D',5:'E',6:'F',7:'T'}
#optimal policy
policy= np.zeros(7)
#set return table as a dictionary
return_table= {(0,1):[],(0,2):[],(0,3):[],(1,4):[],(1,5):[],(2,5):[],(3,5):[],(3,6):[],
				(4,7):[],(5,7):[],(6,7):[]}
#set the q_table
q_table=np.zeros((8,8))
#set the number of episodes
num_episodes= 10000
for i in range(num_episodes):

	print('episode',i)
	#temporary return table for the current episode
	temp_return_table={(0,1):0,(0,2):0,(0,3):0,(1,4):0,(1,5):0,(2,5):0,(3,5):0,(3,6):0,
				(4,7):0,(5,7):0,(6,7):0}
    #current trajectory of the episode
	trajectory = []
	#random initial state
	init_state= random.randint(0,6)
	
	state= np.asarray([init_state])
	
	
	trajectory.append(init_state)
	
	is_terminal = False
	while is_terminal== False:
		#apply epsilon greedy method for generating the action
		probability= random.uniform(0,1)
		if (probability>epsilon):
			action = random.choice(action_table[state[0]])
		else:
			#if current q(s,a) are all zero, choose a random one
			if ((np.amin(q_table[state[0]]))==0):
				action=random.choice(action_table[state[0]])
			# otherwise choose the one with largest non-zero q value
			else:
				print('yes',q_table[state[0]])
				tmp=np.array([q_table[state[0]][x] for x in action_table[state[0]] if x!=0])
				max2=np.max(tmp[np.nonzero(tmp)])
				print('tmp',tmp,max2)
				
				action= list(q_table[state[0]]).index(max2)
				print('action',action)
        #for every state-action pair in the current episode, update the temporary return table
		for key in temp_return_table:
			if temp_return_table[key]!=0:
				#print('reward_table[state[0],action]')
				temp_return_table[key]+=reward_table[state[0],action]
		temp_return_table[state[0],action]+=reward_table[state[0],action]
		
		
		

		
		state[0]=action
			
		trajectory.append(state[0])
		# when reach the terminal episode
		if state[0]== 7:
			print('trajectory',trajectory)
			is_terminal= True
			
			print('trajectory',trajectory)
			#update the q table for each state action pair
			for i in range(len(trajectory)-1):
				if temp_return_table[trajectory[i],trajectory[i+1]]!=0:
					return_table[trajectory[i],trajectory[i+1]].append(temp_return_table[trajectory[i],trajectory[i+1]])
					avg_return=sum(return_table[trajectory[i],trajectory[i+1]])/len(return_table[trajectory[i],trajectory[i+1]])
					#q_table[trajectory[i],trajectory[i+1]]=np.max(return_table[trajectory[i],trajectory[i+1]])
					q_table[trajectory[i],trajectory[i+1]]=q_table[trajectory[i],trajectory[i+1]]+alpha*(np.max(return_table[trajectory[i],trajectory[i+1]])
						-q_table[trajectory[i],trajectory[i+1]])
					
			print('q_table',q_table)
			#update the max action value for the start node
			if (np.amin(q_table[0])==0):
				Source_state.append(0)
			else:
				Source_state.append(np.max(q_table[0][np.nonzero(q_table[0])]))

#for i in range(num_episodes):
#	save_results(Source_state[i],"monte_carlo_alpha_0.01_epsilon_0.1.dat")
			

for i in range(7):
	policy[i]=list(q_table[i]).index(np.max(q_table[i][np.nonzero(q_table[i])]))
print('policy',policy)
plt.figure(1)
plt.plot(Source_state,label="max(Q[0,a])")
plt.xlabel("Episodes")
plt.ylabel("largest action value for Source_state")
plt.legend()
plt.show()

		


	




		




    








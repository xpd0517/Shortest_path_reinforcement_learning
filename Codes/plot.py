import numpy as np
import random 
import matplotlib.pyplot as plt


V1 = open("q_learning_alpha_0.1_epsilon_0.1.dat", "r")
#V2 = open("q_learning_alpha_0.1_epsilon_0.05.dat", "r")
V3 = open("q_learning_alpha_0.1_epsilon_0.01.dat", "r")
V4 = open("q_learning_alpha_0.01_epsilon_0.1.dat", "r")
#V5 = open("q_learning_alpha_0.01_epsilon_0.05.dat", "r")
V6 = open("q_learning_alpha_0.01_epsilon_0.01.dat", "r")
V7 = open("sarsa_alpha_0.1_epsilon_0.1.dat", "r")
#V8 = open("sarsa_alpha_0.1_epsilon_0.05.dat", "r")
V9 = open("sarsa_alpha_0.1_epsilon_0.01.dat", "r")
V10 = open("sarsa_alpha_0.01_epsilon_0.1.dat", "r")
#V11 = open("sarsa_alpha_0.01_epsilon_0.05.dat", "r")
V12 = open("sarsa_alpha_0.01_epsilon_0.01.dat", "r")
V13 = open("monte_carlo_alpha_0.1_epsilon_0.1.dat", "r")
#V14 = open("monte_carlo_alpha_0.1_epsilon_0.05.dat", "r")
V15 = open("monte_carlo_alpha_0.1_epsilon_0.01.dat", "r")
V16 = open("monte_carlo_alpha_0.01_epsilon_0.1.dat", "r")
#V17 = open("monte_carlo_alpha_0.01_epsilon_0.05.dat", "r")
V18 = open("monte_carlo_alpha_0.01_epsilon_0.01.dat", "r")

#V=[V1,V4,V7,V10,V13,V16]
#labels=["q_learning_alpha_0.1_epsilon_0.1","q_learning_alpha_0.01_epsilon_0.1","sarsa_alpha_0.1_epsilon_0.1","sarsa_alpha_0.01_epsilon_0.1"
#			,"monte_carlo_alpha_0.1_epsilon_0.1","monte_carlo_alpha_0.01_epsilon_0.1"]


V=[V3,V6,V9,V12,V15,V18]
labels=["q_learning_alpha_0.1_epsilon_0.01","q_learning_alpha_0.01_epsilon_0.01","sarsa_alpha_0.1_epsilon_0.01","sarsa_alpha_0.01_epsilon_0.01"
			,"monte_carlo_alpha_0.1_epsilon_0.01","monte_carlo_alpha_0.01_epsilon_0.01"]
for i in range(len(labels)):
	list1=[]
	for line in V[i]:
		#print(line.strip("\n"))
		list1.append(float(line.strip("\n")))
	#print("list1",list1)
	plt.plot(list1,label= labels[i])
	plt.xlabel("Episodes")
	plt.ylabel("maxQ(0,a)")
	plt.legend()
plt.show()      
     

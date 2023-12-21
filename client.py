#Aqui vocês irão colocar seu algoritmo de aprendizado
import connection as cn
import random as rd
import numpy as np

socket = cn.connect(2037)

# Parameters
alpha = 0.0 # Learning rate, 0 < alpha < 1, atualizações moderadas
gamma = 0.95 # Discout rate, 0 < gamma < 1, futuro com desconto
epsilon = 0
n_states = 96
n_actions = 3

actions = ("left", "right", "jump")

cur_state = 0
cur_reward = -14

# Criando tabela inicial
def initialize_q_table(n_states, n_actions):
    # Qtable = np.zeros((n_states,n_actions)) #Para criar uma tabela não poluída
    Qtable = np.loadtxt('resultado.txt')
    np.set_printoptions(precision=6)
    return Qtable

# Equação de Bellman para o Q-learning: U(s)=R(s)+γ⋅maxaQ(s′,a)
def utility(next_state, reward, Qtable):
    maxQ = np.max(Qtable[next_state])
    u = reward + gamma * maxQ
    return u

# Política epsilon-greedy
def epsilon_greedy_policy(Qtable, state, epsilon, actions):
    # Exploitation
    if rd.random() > epsilon:
        action_col = np.argmax(Qtable[state])
        # if Qtable[state][action_col] == Qtable[state][2]: # Favorecer jump no início
        #     action_col = 2
        action = actions[action_col]
        print(f'Melhor ação para o estado {state}: {action}')
    # Exploration
    else:
        action_col = int(rd.uniform(0,3))
        action = actions[action_col]
        print(f'A ação aleatória escolhida para o estado {state}: {action}')
    return action, action_col


Q = initialize_q_table(n_states, n_actions)
while True:
    print(f'Estado atual: {cur_state}')

    # De exploration para exploitation gradualmente
    if epsilon > 0.2:
        epsilon -= 0.00001

    action, action_col = epsilon_greedy_policy(Q, cur_state, epsilon, actions)
    state, reward = cn.get_state_reward(socket, action)

    # Pegando só os dígitos de estado e convertendo para decimal
    print(state)
    state = state[2:]
    state = int(state, 2)
    
    print(f'valor anterior da ação: {Q[cur_state][action_col]}')
    Q[cur_state][action_col] = Q[cur_state][action_col] + alpha*(utility(state, reward, Q) - Q[cur_state][action_col])
    print(f'valor novo da ação: {Q[cur_state][action_col]}')

    cur_reward = reward
    cur_state = state

    np.savetxt('resultado.txt', Q, fmt="%f")

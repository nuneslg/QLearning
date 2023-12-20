#Aqui vocês irão colocar seu algoritmo de aprendizado
import connection as cn
import random as rd
import numpy as np

socket = cn.connect(2037)

# Parameters
alpha = 0.7 # Learning rate, 0 < alpha < 1, atualizações moderadas
gamma = 0.9 # Discout rate, 0 < gamma < 1, futuro com desconto
epsilon = 0.5
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
    u = reward + gamma * max(Qtable[next_state])
    return u


# Política epsilon-greedy
def epsilon_greedy_policy(Qtable, state, epsilon, actions):
    random_n = rd.uniform(0,2)
    # Exploitation
    if random_n > epsilon:
        action_col = np.argmax(Qtable[state])
        if Qtable[state][action_col] == Qtable[state][2]:
            action_col = 2
        action = actions[action_col]
        print(f'A ação com maior valor no estado {state} é {action}')
    # Exploration
    else:
        action_col = int(rd.uniform(0,3))
        action = actions[action_col]
        print(f'A ação aleatória escolhida para o estado {state} é {action}')
    return action, action_col



Q = initialize_q_table(n_states, n_actions)

while True:
    print(f'Estado atual: {cur_state}')

    action, action_col = epsilon_greedy_policy(Q, cur_state, epsilon, actions)

    state, reward = cn.get_state_reward(socket, action)

    # Pegando só os dígitos de estado e convertendo para decimal
    state = state[2:]
    state = int(state, 2)
    next_state = state
    
    print(f'valor anterior da ação: {Q[cur_state][action_col]}')
    Q[cur_state][action_col] = Q[cur_state][action_col] + alpha*(utility(next_state, cur_reward, Q) - Q[cur_state][action_col])
    print(f'valor novo da ação: {Q[cur_state][action_col] + alpha*(utility(next_state, cur_reward, Q) - Q[cur_state][action_col])}')

    cur_state = next_state
    cur_reward = reward

    np.savetxt('resultado.txt', Q, fmt="%f")

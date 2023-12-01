import connection as cn

socket = cn.connect(2037)

state, reward = cn.get_state_reward(socket, 'jump') #connection test

print('state: ', state)
print('reward: ', reward)
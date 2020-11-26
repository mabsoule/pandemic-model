

#define initial pandemic state
state_0 = {
    'S':[1000000],
    'I0':[100],
    'Is':[0],
    'Ia':[0],
    'C':[0],
    'H':[0],
    'V':[0]
}

#define budget and spending breakdown X, Y, Z
B = 500
X = 100
Y = 300
Z = 100
print('Valid budget:', B == (X+Y+Z))

#calculate total population and add it to the state_0
N = [sum(i) for i in zip(*list(state_0.values()))]
N_0 = {'N':N}
state_0.update(N_0)

#add iteration limit (aid in troubleshooting)
t_limit = 600 #define maximum iterations before exiting
t_limit = {'t_limit':t_limit}
state_0.update(t_limit)
print('\nInitial parameters:', state_0)

#define pandemic simulator function depending on X, Y, Z
def model(X, Y, Z, state_0):
    #initialize time at day 0
    t = 0

    #initialize state as state_0
    state = state_0

    #initialize weights
    weights = {
        'f_0': [0],
        'v': [0],
        'f_s': [0],
        'f_a': [0],
        'i_s': [0],
        'i_a': [0],
        'c': [0],
        'r_1': [0],
        'r_2': [0],
        'z': [0]
    }

    #enter while loop to calculate iterations until no sick individuals are left or t_limit is hit
    while(t != state.get('t_limit')): #state.get('I0')[t] + state.get('Is')[t] + state.get('Ia')[t] != 0 or
        # print('\niteration:', t)

        #calculates new weight values and append to wieght lists
        weights.get('f_0').append(0.01)
        weights.get('v').append(0.0001)
        weights.get('f_s').append(0.005)
        weights.get('f_a').append(0.001)
        weights.get('i_s').append(0.01)
        weights.get('i_a').append(0.01)
        weights.get('c').append(0.005)
        weights.get('r_1').append(0.01)
        weights.get('r_2').append(0.01)
        weights.get('z').append(int(2.2)) #must be an integer

        # print("weights:", weights)

        #calculate new state values and append to state lists
        state.get('S').append(new_S(state.get('N')[0], state.get('S')[t], state.get('I0')[t], state.get('Is')[t], state.get('Ia')[t], state.get('C')[t], state.get('H')[t], state.get('V')[t], weights.get('f_0')[t], weights.get('v')[t], weights.get('f_s')[t], weights.get('f_a')[t], weights.get('i_s')[t], weights.get('i_a')[t], weights.get('c')[t], weights.get('r_1')[t], weights.get('r_2')[t], weights.get('z')[t]))
        state.get('I0').append(new_I0(state.get('N')[0], state.get('S')[t], state.get('I0')[t], state.get('Is')[t], state.get('Ia')[t], state.get('C')[t], state.get('H')[t], state.get('V')[t], weights.get('f_0')[t], weights.get('v')[t], weights.get('f_s')[t], weights.get('f_a')[t], weights.get('i_s')[t], weights.get('i_a')[t], weights.get('c')[t], weights.get('r_1')[t], weights.get('r_2')[t], weights.get('z')[t]))
        state.get('Is').append(new_Is(state.get('N')[0], state.get('S')[t], state.get('I0')[t], state.get('Is')[t], state.get('Ia')[t], state.get('C')[t], state.get('H')[t], state.get('V')[t], weights.get('f_0')[t], weights.get('v')[t], weights.get('f_s')[t], weights.get('f_a')[t], weights.get('i_s')[t], weights.get('i_a')[t], weights.get('c')[t], weights.get('r_1')[t], weights.get('r_2')[t], weights.get('z')[t]))
        state.get('Ia').append(new_Ia(state.get('N')[0], state.get('S')[t], state.get('I0')[t], state.get('Is')[t], state.get('Ia')[t], state.get('C')[t], state.get('H')[t], state.get('V')[t], weights.get('f_0')[t], weights.get('v')[t], weights.get('f_s')[t], weights.get('f_a')[t], weights.get('i_s')[t], weights.get('i_a')[t], weights.get('c')[t], weights.get('r_1')[t], weights.get('r_2')[t], weights.get('z')[t]))
        state.get('C').append(new_C(state.get('N')[0], state.get('S')[t], state.get('I0')[t], state.get('Is')[t], state.get('Ia')[t], state.get('C')[t], state.get('H')[t], state.get('V')[t], weights.get('f_0')[t], weights.get('v')[t], weights.get('f_s')[t], weights.get('f_a')[t], weights.get('i_s')[t], weights.get('i_a')[t], weights.get('c')[t], weights.get('r_1')[t], weights.get('r_2')[t], weights.get('z')[t]))
        state.get('H').append(new_H(state.get('N')[0], state.get('S')[t], state.get('I0')[t], state.get('Is')[t], state.get('Ia')[t], state.get('C')[t], state.get('H')[t], state.get('V')[t], weights.get('f_0')[t], weights.get('v')[t], weights.get('f_s')[t], weights.get('f_a')[t], weights.get('i_s')[t], weights.get('i_a')[t], weights.get('c')[t], weights.get('r_1')[t], weights.get('r_2')[t], weights.get('z')[t]))
        state.get('V').append(new_V(state.get('N')[0], state.get('S')[t], state.get('I0')[t], state.get('Is')[t], state.get('Ia')[t], state.get('C')[t], state.get('H')[t], state.get('V')[t], weights.get('f_0')[t], weights.get('v')[t], weights.get('f_s')[t], weights.get('f_a')[t], weights.get('i_s')[t], weights.get('i_a')[t], weights.get('c')[t], weights.get('r_1')[t], weights.get('r_2')[t], weights.get('z')[t]))

        # print("state:", state)

        #iterate time period by 1 day
        t +=1

    #notify if simulation failed to finish before t hit t_limit
    if(t == state.get('t_limit')):
        print('t_limit hit')

    return state.get('S'), state.get('I0'), state.get('Is'), state.get('Ia'), state.get('C'), state.get('H'), state.get('V')


#define state functions

def phi(z):
    phi = 2*z*(z+1)
    return phi

def new_S(N, S, I0, Is, Ia, C, H, V, f_0, v, f_s, f_a, i_s, i_a, c, r_1, r_2, z):
    S_delta = -1 * phi(z) * (S/N) * (I0*f_0 + Is*f_s + Ia*f_a) - S*v
    new_S = S + S_delta
    return new_S

def new_I0(N, S, I0, Is, Ia, C, H, V, f_0, v, f_s, f_a, i_s, i_a, c, r_1, r_2, z):
    I0_delta = phi(z) * (S/N) * (I0*f_0 + Is*f_s + Ia*f_a) - I0*i_s - I0*i_a
    new_I0 = I0 + I0_delta
    return new_I0

def new_Is(N, S, I0, Is, Ia, C, H, V, f_0, v, f_s, f_a, i_s, i_a, c, r_1, r_2, z):
    Is_delta = I0*i_s - Is*c - Is*r_1
    new_Is = Is + Is_delta
    return new_Is

def new_Ia(N, S, I0, Is, Ia, C, H, V, f_0, v, f_s, f_a, i_s, i_a, c, r_1, r_2, z):
    Ia_delta = I0*i_a - Ia*r_2
    new_Ia = Ia + Ia_delta
    return new_Ia

def new_C(N, S, I0, Is, Ia, C, H, V, f_0, v, f_s, f_a, i_s, i_a, c, r_1, r_2, z):
    C_delta = Is*c
    new_C = C + C_delta
    return new_C

def new_H(N, S, I0, Is, Ia, C, H, V, f_0, v, f_s, f_a, i_s, i_a, c, r_1, r_2, z):
    H_delta = Is*r_1 + Ia*r_2
    new_H = H + H_delta
    return new_H

def new_V(N, S, I0, Is, Ia, C, H, V, f_0, v, f_s, f_a, i_s, i_a, c, r_1, r_2, z):
    V_delta = S*v
    new_V = V + V_delta
    return new_V


#call pandemic simulator function
S, I0, Is, Ia, C, H, V = model(X, Y, Z, state_0)
print(C)

#plot results
import matplotlib.pyplot as plt
import numpy as np
from operator import add

plt.close('all')
plt.figure(1)
t = np.arange(0,len(S),1)
plt.plot(t, S, label='Susceptible', color='k')
plt.plot(t, I0, label='Infected, no symptomes', color='m')
plt.plot(t, Is, label='Infected, symptomatic', color='c',)
plt.plot(t, Ia, label='Infected, asymptomatic', color='b')
plt.plot(t, C, label='Casualties', color='r')
plt.plot(t, H, label='Recovered', color='y')
plt.plot(t, V, label='Vacinated + Natural Immunity', color='g')
plt.legend()
plt.title('Pandemic model')
plt.xlabel('Time (days)')
plt.ylabel('Number of People')

#simple plot
# plt.figure(2)
# t = np.arange(0,len(S),1)
# plt.plot(t, S, label='Susceptible')
# plt.plot(t, list(map(add, map(add, I0, Is), Ia)), label='Total Infected')
# plt.plot(t, C, label='Casualties')
# plt.legend()
# plt.title('Pandemic model')
# plt.xlabel('Time (days)')
# plt.ylabel('Number of People')
plt.show()

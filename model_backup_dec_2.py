import numpy as np

#define initial pandemic state
state_0 = {
    'S':[100000],
    'I0':[1],
    'Is':[0],
    'Ia':[0],
    'C':[0],
    'H':[0],
    'V':[0]
}

#define budget and spending breakdown X, Y, Z
#200 (Y) million vacine
#30 (X) million for awareness
#? (Z) for hospitalization

# Values are in 10^6 Canadian Dollars
B = 0.75 * (50+200+1000)
X = 50 #0-50
Y = 200 #0-200
Z = 1000 #0-1000
print('Valid budget:', B == (X+Y+Z))

#calculate total population and add it to the state_0
N = [sum(i) for i in zip(*list(state_0.values()))]
N_0 = {'N':N}
state_0.update(N_0)

#add iteration limit (aid in troubleshooting)
t_limit = 1200 #define maximum iterations before exiting
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
        weights.get('f_0').append(0.008) #Sachin
        weights.get('v').append(new_v(t, Y))
        weights.get('f_s').append(0.01) #Sachin
        weights.get('f_a').append(0.008) #Sachin
        weights.get('i_s').append(new_i_s())
        weights.get('i_a').append(new_i_a())
        weights.get('c').append(0.005) #Christos
        weights.get('r_1').append(0.01) #Christos
        weights.get('r_2').append(1/14) #Christos
        weights.get('z').append(new_z(t, X)) #Avery

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

    return state.get('S'), state.get('I0'), state.get('Is'), state.get('Ia'), state.get('C'), state.get('H'), state.get('V'), weights.get('f_0'), weights.get('v'), weights.get('f_s'), weights.get('f_a'), weights.get('i_s'), weights.get('i_a'), weights.get('c'), weights.get('r_1'), weights.get('r_2'), weights.get('z')


#define state functions

# def phi(z):
#     phi = 2*z*(z+1)
#     return phi

def new_S(N, S, I0, Is, Ia, C, H, V, f_0, v, f_s, f_a, i_s, i_a, c, r_1, r_2, z):
    S_delta = -1 * z * (S/N) * (I0*f_0 + Is*f_s + Ia*f_a) - S*v
    new_S = S + S_delta
    return new_S

def new_I0(N, S, I0, Is, Ia, C, H, V, f_0, v, f_s, f_a, i_s, i_a, c, r_1, r_2, z):
    I0_delta = z * (S/N) * (I0*f_0 + Is*f_s + Ia*f_a) - I0*i_s - I0*i_a
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

# define weight functions
def new_i_s():
    i_s = 0.8 / 6 #80% of people develop symptomes after on average 6 days
    return i_s

def new_i_a():
    i_a = 0.2 / 6 #80% of people develop symptomes after on average 6 days
    return i_a

def new_v(t, Y):
    if(Y>200): #Spending over a certain limit no longer speeds up the process due to time taken for clinical trials
        Y = 200
    spending_shift = -1*(18/5*Y) + 1080
    v = 0.5196*np.exp(0.0146*(t - spending_shift))/100 #divide by 100 to convert percentage to decimal
    if(v>1):
        v = 1
    return v

def new_z(t, X): #note deleting phi as our model can more fluidly adjust to changes in funding rather than steped result when using phi(z)
    Z = (6.7/50)*X
    if(Z > 6.7):
        Z = 6.7
    z = 13.4 + Z*np.exp((-1/13)*t) - Z
    return z


#call pandemic simulator function
S, I0, Is, Ia, C, H, V, f_0, v, f_s, f_a, i_s, i_a, c, r_1, r_2, z = model(X, Y, Z, state_0)
print(C)

#plot results
import matplotlib.pyplot as plt
import numpy as np
from operator import add

#plot state variables
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


#plot weight variables
plt.figure(2)
t = np.arange(0,len(z),1)
plt.plot(t, f_0, label='f_0')
plt.plot(t, v, label='v')
plt.plot(t, f_s, label='f_s')
plt.plot(t, f_a, label='f_a')
plt.plot(t, i_s, label='i_s')
plt.plot(t, i_a, label='i_a')
plt.plot(t, c, label='c')
plt.plot(t, r_1, label='r_1')
plt.plot(t, r_2, label='r_2')
plt.plot(t, z, label='z')
plt.legend()
plt.title('Pandemic model weights')
plt.xlabel('Time (days)')
plt.ylabel('Probability weights')

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

# plt.figure(2)
# t = np.arange(0,720,1)
# v = new_v(t, Y)
# plt.plot(t, v)

plt.show()

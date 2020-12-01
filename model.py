import numpy as np
import sympy as sym
from scipy.optimize import minimize

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
# B = 0.75 * (50+200+1000)
# X = 50 #0-50
# Y = 200 #0-200
# Z = 1000 #0-1000
# print('Valid budget:', B == (X+Y+Z))

#define pandemic simulator function depending on X, Y, Z
def model(money, state_0, flag=False):
    X = money[0]
    Y = money[1]
    Z = money[2]
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
        weights.get('f_0').append(new_f_0(X)) #Sachin
        weights.get('v').append(new_v(t, Y))
        weights.get('f_s').append(new_f_s(t, X, Z)) #Sachin
        weights.get('f_a').append(new_f_a(t, X, Z)) #Sachin
        weights.get('i_s').append(new_i_s())
        weights.get('i_a').append(new_i_a())
        weights.get('c').append(0.005) #Christos
        weights.get('r_1').append(0.01) #Christos
        weights.get('r_2').append(0.01) #Christos
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

    if flag==True:
        return max(state.get('C'))
    else:
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

def new_f_0(X):
  x = sym.Symbol('x')
  budget1 = 0 # in millions
  f0_prob1 = 20/100 # upper bound of probability in decimal 
  budget2 = 50 # change to max X budget for flexbility
  f0_prob2 = 0.01/100 # lower bound of probability in decimal 
  m = (f0_prob2 - f0_prob1) / (budget2 - budget1)
  f0 = m*(x-budget1)+f0_prob1
  return f0.subs({x:X})

def new_f_s(t, X, Z):
  x = sym.Symbol('x')
  z = sym.Symbol('z')
  
  avgSpreadRate = 0.15
  
  xBudget1 = 0 # change to low X budget for flexbility
  zBudget1 = 770 # change to low Z budget for flexbility
  fs_prob1 = (avgSpreadRate+0.1) # upper bound of spread probability when spending lower (in decimal) 

  xBudget2 = 50 # change to high X budget for flexbility
  zBudget2 = 1000 # change to high Z budget for flexbility
  fs_prob2 = (avgSpreadRate-0.1) # lower bound of spread probability 

  # Symbolic equations below that adapt in case upper/lower bounds change
  m_x = (fs_prob2 - fs_prob1) / (xBudget2 - xBudget1)
  fs_x =  m_x*(x-xBudget1)+fs_prob1

  m_z = (fs_prob2 - fs_prob1) / (zBudget2 - zBudget1)
  fs_z =  m_z*(z-zBudget1)+fs_prob1

  if t <= 180: # Assuming weight of awareness spending is significant in first 6 months
    wx = 0.6
  else:
    wx = 0.1
    
  fs = wx*fs_x.subs({x:X}) + (1-wx)*fs_z.subs({z:Z})
  return fs

def new_f_a(t, X, Z):
  fa = (1-0.42)*new_f_s(t, X, Z) # asymptomatic probabiliy is 42% less than symptomatic
  return fa


#plot results
import matplotlib.pyplot as plt
import numpy as np
from operator import add

X = 45 #0-50
Y = 100 #0-200
Z = 780 #770-1000
budget = [X, Y, Z]

#calculate total population and add it to the state_0
N = [sum(i) for i in zip(*list(state_0.values()))]
N_0 = {'N':N}
state_0.update(N_0)

#add iteration limit (aid in troubleshooting)
t_limit = 600 #define maximum iterations before exiting
t_limit = {'t_limit':t_limit}
state_0.update(t_limit)
print('\nInitial parameters:', state_0)

# optimization
def constraint1(initial_values):
    X = initial_values[0]
    Y = initial_values[1]
    Z = initial_values[2]
    return 1250-X-Y-Z
    #return X+Y+Z-1250

x0 = [50,200,1000] # initial points

# bounds for X, Y, Z
X_bound = (0,50)
Y_bound = (0,200)
Z_bound = (770,1000)

bnds = (X_bound, Y_bound, Z_bound)
con1 = {'type':'eq', 'fun':constraint1} # currently set as equality constraint for ineqaulity use: ineq

# C1 = model(budget, state_0, True)
# print(C1)

# sol = minimize(model, x0, method='SLSQP', bounds=bnds, constraints=con1, args=(state_0, True)) 
# print(sol)

# C2 = model(budget, state_0, True)
# print(budget, C2)

#call pandemic simulator function
S, I0, Is, Ia, C, H, V, f_0, v, f_s, f_a, i_s, i_a, c, r_1, r_2, z = model(budget, state_0)
print(C)
##plot state variables
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

##plot weight variables
# plt.figure(2)
# t = np.arange(0,len(z),1)
# plt.plot(t, f_0, label='f_0')
# # plt.plot(t, v, label='v')
# plt.plot(t, f_s, label='f_s')
# plt.plot(t, f_a, label='f_a')
# # plt.plot(t, i_s, label='i_s')
# # plt.plot(t, i_a, label='i_a')
# # plt.plot(t, c, label='c')
# # plt.plot(t, r_1, label='r_1')
# # plt.plot(t, r_2, label='r_2')
# # plt.plot(t, z, label='z')
# plt.legend()
# plt.title('Pandemic model weights')
# plt.xlabel('Time (days)')
# plt.ylabel('Probability weights')

##simple plot
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

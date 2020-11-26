

#call function to
S, I0, Is, Ia, C, H, V = model(X, Y, Z)

#initial number of people
state = {
    'S':[10000],
    'I0':[1],
    'Is':[0],
    'Ia':[0],
    'C':[0],
    'H':[0],
    'V':[0]
}

# iteration -> 1 day = 1 iteration
t = 0
N = S + I0 + Is + Ia + C + H + V

#initial weights / probabilities
weights = {
    'f_o': [0.1], #S
    'v': [0.1], #S
    'f_s': [0.1], #S
    'f_a': [0.1], #A
    'i_s': [0.1], #A
    'i_a': [0.1], #A
    'c': [0.1], #C
    'r_1': [0.1], #C
    'r_2': [0.1] #C
}

while(I0 + Is + Ia != 0)

    #calculates new weight variables
    new_weights = {
        'f_o': weights('f_0').append(new_f_o()[t]),
        'v': new_f_o(),
        'f_s': new_f_o(),
        'f_a': new_f_o(),
        'i_s': new_f_o(),
        'i_a': new_f_o(),
        'c': new_f_o(),
        'r1': new_f_o(),
        'r2': new_f_o(),
        'z': new_z()
    }

    #calculate new state
    new_state = {
        'S': new_S(),
        'I0': new_I0(),
        'Is': new_Is(),
        'Ia': new_Ia(),
        'C': new_C(),
        'H': new_H(),
        'V': new_V(),
    }
    weights.update(new_weights)
    state.update(new_state)

    #update iteration count
    t += 1


def new_S():
    change = -phi(z) * (S/N) * (I0*f_o + Is*f_s + Ia*f_a) - S*v #change syntax to key value pair
    new_S = old_S + change
    return new_S


def f_s(X, Z, t):
    probability = 1 - X_prob(t, X) - Z_prob(t, Z)

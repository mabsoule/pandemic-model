

state = {
    'suseptable':10000,
    'infected':0,
    'infected_symtomatic':0,
    'infected_asymtomatic':0,
    'causualty':0,
    'recovered':0,
    'immune':0,
}

initial_weights = {

}

while()

    #calculates new weight variables
    new_weights = get_weights()

    #calculate new state
    new_state = {
        'suseptable': new_suseptable(),
        'infected': new_infected(),
        'infected_symtomatic': new_infected_symtomatic(),
        'infected_asymtomatic': new_infected_asymtomatic(),
        'causualty': new_casualty(),
        'recovered': new_recovered(),
        'immune': new_immune(),
    }
    state.update(new_state)

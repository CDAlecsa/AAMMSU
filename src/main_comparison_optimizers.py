###############################################
#                 MODULES
###############################################
import torch, os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from torch.optim import Adam

from AAMMSU_optim import AAMMSU
from various_optimizers.Padam_optim import Padam
from various_optimizers.AdaBelief_optim import AdaBelief
from various_optimizers.Apollo_optim import Apollo
from various_optimizers.Ranger_optim import Ranger
from various_optimizers.Madgrad_optim import MADGRAD




###############################################
#             OBJECTIVE FUNCTIONS
###############################################
def objective_function(data, fct = 'Rosenbrock', lib = torch):
    x, y = data
    if fct == 'hybrid norm':
        return lib.sqrt(1 + x ** 2) + lib.sqrt(1 + y ** 2)
    elif fct == 'Rosenbrock':
        return (1 - x) ** 2 + 100 * (y - x**2) ** 2
    elif fct == 'polynomial':
        return  (x + y) ** 4 + (x/2 - y/2) ** 4 
    



###############################################
#             INITIAL CONDITIONS
###############################################
def get_initial_conditions(fct = 'Rosenbrock'):
    if fct == 'hybrid norm':
        initial_state = (1.0, -2.0)
    elif fct == 'Rosenbrock':
        initial_state = (-2.0, 2.0)
    elif fct == 'polynomial':
        initial_state = (0.5, -2.5)
    return initial_state



###############################################
#                 BOUNDS
###############################################
def get_bounds(fct = 'Rosenbrock'):
    if fct == 'hybrid norm':
        x = np.linspace(-1.0, 1.5, 250)
        y = np.linspace(-3.0, 1.0, 250)
    elif fct == 'Rosenbrock':
        x = np.linspace(-2.3, 2, 250)
        y = np.linspace(-1, 3.5, 250)
    elif fct == 'polynomial':
        x = np.linspace(-1.5, 3.0, 250)
        y = np.linspace(-3.0, 0.5, 250)
    return x, y



###############################################
#                 MINIMUM POINTS
###############################################
def get_minimum(fct = 'rosenbrock'):
    if fct == 'hybrid norm':
        return (0.0, 0.0)
    elif fct == 'Rosenbrock':
        return (1.0, 1.0)
    elif fct == 'polynomial':
        return (0.0, 0.0)



###############################################
#                 TRAIN FUNCTION
###############################################
def execute_steps(func, initial_state, x, optimizer, num_iter = 500):
    steps = []
    steps = np.zeros((2, num_iter + 1))
    steps[:, 0] = np.array(initial_state)
    for i in range(1, num_iter + 1):
        optimizer.zero_grad()
        f = func(x)
        f.backward()
        torch.nn.utils.clip_grad_norm_(x, 1.0)
        optimizer.step()
        steps[:, i] = x.detach().numpy()
    return steps





###############################################
#                 PLOT TRAJECTORIES
###############################################
def plot_objective_function(steps_dict, function_type, experiment_name):

    fig = plt.figure(figsize = (16, 8))

    n_cols = len(function_type)
    cmap = ['black', 'blue', 'red', 'darkorange', 'yellow', 'brown', 'gray']

    for count, f in enumerate(function_type):
        ax = fig.add_subplot(1, n_cols, count + 1)

        x, y = get_bounds(fct = f)
        minimum = get_minimum(fct = f)
        X, Y = np.meshgrid(x, y)
        Z = objective_function([X, Y], fct = f, lib = np)
        
        ax.contour(X, Y, Z, 20, cmap = "jet")



        for optim_count, (optimizer_name, steps) in enumerate(steps_dict[f].items()):
            iter_x, iter_y = steps[0, :], steps[1, :]
            
            ax.plot(iter_x, iter_y, color = cmap[optim_count], marker = "o", label = optimizer_name, alpha = 0.8)
            ax.plot(iter_x[0], iter_y[0], "r*", markersize = 8)
            ax.plot(iter_x[-1], iter_y[-1], "gs", markersize = 8)
            ax.plot(*minimum, "kD", markersize = 8)
            
        ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
        ax.tick_params(axis = 'both', which = 'minor', labelsize = 10)

        ax.legend(loc = 'upper right', fontsize = 15)
        ax.set_title(f, fontsize = 20)

    plt.tight_layout()

    os.makedirs('./optimizer_comparison_results/', exist_ok = True)
    plt.savefig('./optimizer_comparison_results/various_optimizers_' + experiment_name + '.jpeg', dpi = 300)

    plt.show()




###############################################
#                 EXPERIMENTS
###############################################
def execute_experiments_1_and_2(optimizer_names, function_type, num_iter, lr, betas, eps, experiment_name):
    
    steps_func = {f : {name : None for name in optimizer_names} for f in function_type}

    for f in function_type:

        initial_state = get_initial_conditions(f)
        fct = lambda var : objective_function(var, f)

        for name in optimizer_names:
            x = torch.Tensor(initial_state).requires_grad_(True)
            
            if name == 'AAMMSU':
                optim = AAMMSU([x], lr = lr, M = 0.75, mu = 0.5, nu = 0.5, tilde_gamma = 0.75, beta_2 = betas[1], eps = eps)
            elif name == 'Madgrad':
                optim = MADGRAD([x], lr = lr, momentum = 0.9, eps = eps)
            elif name == 'Adam':
                optim = Adam([x], lr = lr, betas = betas)
            elif name == 'Padam':
                optim = Padam([x], lr = lr, betas = betas, eps = eps, amsgrad = True, partial = 1/4)
            elif name == 'AdaBelief':
                optim = AdaBelief([x], lr = lr, betas = betas, eps = eps, amsgrad = True, degenerated_to_sgd = True)
            elif name == 'Apollo':
                optim = Apollo([x], lr = lr, beta = betas[0], eps = eps, warmup = 500)
            elif name == 'Ranger':
                optim = Ranger([x], lr = lr, alpha = 0.5, k = 6, N_sma_threshhold = 5, betas = betas, eps = eps)

            steps_func[f][name] = execute_steps(fct, initial_state, x, optim, num_iter)


    plot_objective_function(steps_func, function_type, experiment_name)






def execute_experiment_3(optimizer_names, function_type, num_iter, lr, betas, eps, experiment_name):
    
    steps_func = {f : {name : None for name in optimizer_names} for f in function_type}

    for f in function_type:

        initial_state = get_initial_conditions(f)
        fct = lambda var : objective_function(var, f)

        for name in optimizer_names:
            x = torch.Tensor(initial_state).requires_grad_(True)            
            coeffs = eval(name)
            optim = AAMMSU([x], lr = lr, M = coeffs[0], mu = coeffs[1], 
                           nu = coeffs[2], tilde_gamma = coeffs[3], beta_2 = betas[1], eps = eps)

            steps_func[f][name] = execute_steps(fct, initial_state, x, optim, num_iter)


    plot_objective_function(steps_func, function_type, experiment_name)






###############################################
#                 MAIN
###############################################

if __name__ == "__main__":
    betas = (0.9, 0.999)
    eps = 1e-8

    func = ['Rosenbrock', 'polynomial', 'hybrid norm']

    optimizers_for_exp_1_and_2 = ['AAMMSU', 'Adam', 'Padam', 'AdaBelief', 'Apollo', 'Ranger', 'Madgrad']         
    optimizers_for_exp_3 = ['(0.75, 0.5, 0.5, 0.75)', '(1.25, 0.5, 0.5, 0.75)', '(0.25, 0.75, 0.5, 0.75)',
                            '(0.25, 0.5, 0.9, 0.75)', '(0.25, 0.5, 0.9, 0.95)']
    

    np.random.seed(10)
    execute_experiments_1_and_2(optimizers_for_exp_1_and_2, func, 100, 1e-1, betas, eps, 'exp_1')
    execute_experiments_1_and_2(optimizers_for_exp_1_and_2, func, 400, 1e-3, betas, eps, 'exp_2')
    execute_experiment_3(optimizers_for_exp_3, func, 100, 1e-1, betas, eps, 'exp_3')




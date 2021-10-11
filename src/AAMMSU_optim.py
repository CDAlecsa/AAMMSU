################################
#               MODULES
################################
import torch
from torch.optim import Optimizer







######################################
#             AAMMSU OPTIMIZER
######################################
class AAMMSU(Optimizer) :
    
    # Constructor
    def __init__(self, params, lr = 1e-3, M = 2, mu = 0.5, nu = 0.5, tilde_gamma = 0.5, beta_2 = 0.999, eps = 1e-8) :

        # Specific invalid cases for the parameters
        if not 0.0 <= lr :
            raise ValueError("learning rate {} must be non-negative".format(lr))
        if not 0.0 < M :
            raise ValueError("M {} must be positive".format(M))
        if not 0.0 < mu < 1 :
            raise ValueError("mu value {} must be in (0, 1)".format(mu))
        if not 0.0 < nu < 1 :
            raise ValueError("nu value {} must be in (0, 1)".format(nu))
        if not mu <= tilde_gamma < 1 :
            raise ValueError("tilde_gamma value {} must be in [mu, 1) = ({}, 1)".format(tilde_gamma, mu))
        if not 0.0 < beta_2 < 1 :
            raise ValueError("beta_2 value {} must be in (0, 1)".format(beta_2))    
        if not 0.0 < eps < 1 :
            raise ValueError("epsilon value {} must be in (0, 1)".format(eps))
        

        # Update the constructor
        defaults = dict(lr = lr, M = M, mu = mu, nu = nu, tilde_gamma = tilde_gamma, beta_2 = beta_2, eps = eps)
        super(AAMMSU, self).__init__(params, defaults)

        
        
    # The step function related to the Optimizer class
    def step(self, closure = None) :
 
        # Retrieve the loss function
        loss = None
        if closure is not None:
            loss = closure()

            
        # Loop over the group of parameters
        for group in self.param_groups :
            
            # Retrieve the coefficients of the optimizer
            lr, M, mu, nu, tilde_gamma, beta_2, eps = (group['lr'], group['M'], group['mu'], group['nu'], group['tilde_gamma'], group['beta_2'], group['eps'])

            # Loop of the main parameters, i.e. weights & biases
            for p in group['params'] :
                
                # Pass if we don't have gradients for the current parameters
                if p.grad is None :
                    continue
                
                # Take the current gradient
                grad = p.grad.data
                
                # The case when we have sparse gradients is not available
                if grad.is_sparse :
                    msg = ('AAMMSU does not support sparse gradients !')
                    raise RuntimeError(msg)

                # Retrieve the state for the current iteration
                state = self.state[p]

                # TWhen n = 0, i.e. for the 1st iteration, we set the initial values
                if len(state) == 0 :
                    
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(p, memory_format = torch.preserve_format)
                    
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format = torch.preserve_format)
                    state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format = torch.preserve_format)
                    
                    state['previous_alpha_n'] = torch.zeros_like(p, memory_format = torch.preserve_format)
                    state['previous_grad'] = torch.zeros_like(p, memory_format = torch.preserve_format)
                        

                # Memorize the values of the state parameters
                tilde_v, v, m, alpha_n_prev, grad_prev = (state['exp_avg_sq'], state['max_exp_avg_sq'], state['momentum'], state['previous_alpha_n'], state['previous_grad'])              
                
                # Update the value of the current iteration
                state['step'] += 1
                
                # Compute the AMSGrad-type exponential moving average of the squared gradients
                tilde_v.mul_(beta_2).addcmul_(grad, grad, value = 1 - beta_2)
                torch.maximum(v, tilde_v, out = v)
                
                # Compute the effective stepsize
                a_n = lr / ( eps + v.sqrt() )
                
                # Compute the adaptive stepsize
                alpha_n = nu * a_n

                
                # Compute the actual algorithm
                if state['step'] == 2 :
                    
                    # When n = 2, compute the shifted variable z_n
                    p.data.addcmul_( alpha_n, grad, value = - (1 + tilde_gamma * (M - 1) ) )
                    
                    
                elif state['step'] >= 3 :
                    
                    # When n = 3, compute real-valued coeffients
                    beta_n_prev = self.compute_beta_n(state['step'] - 1, mu)
                
                    mu_n_prev = self.compute_mu_n(state['step'] - 1, mu)
                    mu_n_prev_2 = self.compute_mu_n(state['step'] - 2, mu)
                    
                    gamma_n = self.compute_gamma_n(state['step'], tilde_gamma, mu)
                    gamma_n_prev = self.compute_gamma_n(state['step'] - 1, tilde_gamma, mu)
                    
                    tilde_gamma_n = self.compute_tilde_gamma_n(state['step'], tilde_gamma)
                    tilde_gamma_n_prev = self.compute_tilde_gamma_n(state['step'] - 1, tilde_gamma)
                    
                    # Update the shifted variable z_n
                    p.data.add_(m, alpha = beta_n_prev * (1 + gamma_n) - gamma_n_prev ).addcmul_(grad_prev, alpha_n_prev, value = - (M * mu_n_prev_2 - 1) * (mu_n_prev * (1 + gamma_n) - tilde_gamma_n_prev) / mu_n_prev_2 ).addcmul_(grad, alpha_n, value = - ( (1 + gamma_n) + tilde_gamma_n * (M * mu_n_prev - 1) / mu_n_prev ) )
                    
                    # Update the momentum term m_n
                    m.mul_(beta_n_prev).addcmul_(grad_prev, alpha_n_prev, value = - mu_n_prev * (M * mu_n_prev_2 - 1) / mu_n_prev_2 ).addcmul_(grad, alpha_n, value = - 1)

                    
                # Memorize the previous value of the adaptive stepsize
                state['previous_alpha_n'] = alpha_n.clone()
                
                # Memorize the previous value of the gradient
                state['previous_grad'] = grad.clone()

        # Return the value of the loss function
        return loss
    
    
    
    
    # Custom function for computing real-valued coefficients 
    @staticmethod
    def compute_beta_n(n, given_mu) :
        if n == 2 :
            return 0
        elif n >= 3 :
            return 1 - given_mu
        
    @staticmethod
    def compute_mu_n(n, given_mu) :
        if n == 1 :
            return 1
        elif n >= 2 :
            return given_mu
    
    @staticmethod
    def compute_tilde_gamma_n(n, given_tilde_gamma) :
        if n == 1 :
            return 1
        elif n >= 2 :
            return given_tilde_gamma
        
    @staticmethod
    def compute_gamma_n(n, given_tilde_gamma, given_mu) :
        if n == 2 :
            return 0
        elif n >= 3 :
            return given_tilde_gamma * (1 - given_mu) / given_mu
    
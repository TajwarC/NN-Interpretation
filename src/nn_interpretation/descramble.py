import numpy as np
from scipy.optimize import minimize
from scipy.linalg import toeplitz, inv, norm

def get_fourdif_2nd_order(N):
    """
    Generates the 2nd order Fourier spectral differentiation matrix.
    Recreates the core functionality of MATLAB's fourdif(N, 2)
    based on standard spectral methods.
    """
    c = np.zeros(N)
    
    # Construct the first column of the Toeplitz matrix
    if N % 2 == 0:
        c[0] = -(N**2 + 2) / 12.0
        k = np.arange(1, N)
        c[1:] = 0.5 * ((-1)**k) / (np.sin(k * np.pi / N)**2)
    else:
        c[0] = -(N**2 - 1) / 12.0
        k = np.arange(1, N)
        c[1:] = 0.5 * ((-1)**k) * np.cos(k * np.pi / N) / (np.sin(k * np.pi / N)**2)
    
    # D is a symmetric Toeplitz matrix for the 2nd derivative
    D = toeplitz(c)
    return D

def descramble(S, n_iter=400, guess=None):
    """
    Generates a weight matrix descrambler for a particular layer in
    a neural network using Tikhonov smoothness criterion.
    
    Parameters:
        S (np.ndarray): A matrix containing, in its columns, the outputs
                        of the preceding layers of the neural network for
                        a (preferably large) number of reasonable inputs.
        n_iter (int): Maximum number of L-BFGS-B iterations. Default is 400.
        guess (np.ndarray): [optional] The initial guess for the descrambling
                            transform generator. Default is a zeros matrix.
                            
    Returns:
        P (np.ndarray): Descrambling matrix.
        Q (np.ndarray): Generator matrix.
    """
    # Check consistency
    if not isinstance(S, np.ndarray) or not np.isrealobj(S):
        raise ValueError("S must be a real numpy array.")
    if not isinstance(n_iter, int) or n_iter < 1:
        raise ValueError("n_iter must be a positive integer.")
        
    out_dim = S.shape[0]
    opt_dim = (out_dim**2 - out_dim) // 2
    
    # Lower triangle index array
    lt_idx = np.tril_indices(out_dim, -1)
    
    # Default guess is zero
    if guess is None:
        q0 = np.zeros(opt_dim)
    else:
        q0 = guess[lt_idx]
        
    # Get second derivative operator
    D = get_fourdif_2nd_order(out_dim)
    
    # Precompute some steps and scale
    SST = S @ S.T
    SST = out_dim * SST / norm(SST, 2)
    DTD = D.T @ D
    DTD = out_dim * DTD / norm(DTD, 2)
    U = np.eye(out_dim)
    
    # Regularisation signal (Objective Function)
    def reg_sig(q):
        # Form the generator
        Q = np.zeros((out_dim, out_dim))
        Q[lt_idx] = q
        Q = Q - Q.T
        
        # Re-use the inverse
        iUpQ = inv(U + Q)
        
        # Run Cayley transform
        P = iUpQ @ (U - Q)
        
        # Re-use triple product
        DTDPSST = DTD @ P @ SST
        
        # Compute Tikhonov norm
        eta = np.trace(DTDPSST @ P.T)
        
        # Compute Tikhonov norm gradient
        eta_grad = -2.0 * iUpQ.T @ DTDPSST @ (U + P).T
        
        # Antisymmetrise the gradient
        eta_grad = eta_grad - eta_grad.T
        
        # Extract the lower triangle
        eta_grad_1d = eta_grad[lt_idx]
        
        return eta, eta_grad_1d

    # Optimisation using L-BFGS-B 
    options = {
        'maxiter': n_iter,
        'disp': True, # Set to False to silence solver output
    }
    
    # Run the bounded memory BFGS minimizer
    res = minimize(reg_sig, q0, method='L-BFGS-B', jac=True, options=options)
    q_opt = res.x
    
    # Form descramble generator from the optimized values
    Q = np.zeros((out_dim, out_dim))
    Q[lt_idx] = q_opt
    Q = Q - Q.T
    
    # Run final Cayley transform
    P = inv(U + Q) @ (U - Q)
    
    return P, Q
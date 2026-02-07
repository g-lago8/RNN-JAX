"""
Parallelized through time implementation of a vanilla rnn
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from optax.losses import squared_error


def diag_jacobian(f, args, argnum=0, output_idx=0):
    """Compute diagonal of Jacobian for specific argument and output."""
    x = args[argnum]
    eye = jnp.eye(len(x))
    
    def f_specific(x_var):
        args_list = list(args)
        args_list[argnum] = x_var
        outputs = f(*args_list)
        return outputs[output_idx]
    
    _, jvp_products = jax.vmap(lambda v: jax.jvp(f_specific, (x,), (v,)))(eye)
    return jvp_products



def sequential(step_fn, u_sequence, initial_state):
    """Sequential through time implementation of a vanilla rnn"""
    x = initial_state
    f = lambda x, u : step_fn(u, x) # f(x_t, u_t+1) = x_t+1
    _, xs = jax.lax.scan(f, x, u_sequence) # call the scan fn
    return xs


def operator_diag(op1, op2):
    """Solves a step of a linear recurrence $x_{t+1} = A_{t+1}x_{t} + b$
    """
    A1, b1 = op1
    A2, b2 = op2
    return A1 * A2, A2 * b1 + b2


def operator_add(op1, op2):
    return op1 + op2

def operator_full(op1, op2):
    A1, b1 = op1
    A2, b2 = op2
    # A1 and A2 are (state_dim, state_dim) matrices
    # b1 and b2 are (state_dim,) vectors
    # We compute the composition of two affine transforms:
    # x -> A1 @ x + b1 followed by x -> A2 @ x + b2
    # Result: x -> A2 @ (A1 @ x + b1) + b2 = (A2 @ A1) @ x + (A2 @ b1 + b2)
    return jnp.matmul(A2, A1), jnp.matvec(A2, b1)+ b2


def initial_guess(u_sequence, initial_state, key):
    key, subkey = jr.split(key)
    state_dim = initial_state.shape[0]
    seq_len = u_sequence.shape[0]
    initial_guess = jr.normal(subkey, (seq_len+1, state_dim)) # initial guess of our trajectory x_0, ..., x_T
    x_i = initial_guess.at[0].set(initial_state)
    return key, x_i


def merit_function(old_extimates, new_extimates):
    """squared 2-norm of the difference between new and old estimates

    Args:
        old_extimates (jax.numpy.array): old extimates, of lenght T
        new_extimates (jax.numpy.array): new extimates, of lenght T+1

    Returns:
        _type_: _description_
    """
    return 1/2 *  squared_error(old_extimates-new_extimates).mean()   


def parallel_jacobi(step_fn, u_sequence, initial_state, eps=1e-15, max_steps=10, *, key):
    
    key, x_i = initial_guess(u_sequence, initial_state, key) 
    step = 0
    while True: # iterate guessing the whole sequence, until the guess is good enough
        step += 1
        f = lambda x, u : step_fn(u, x)[0] # f(x_t, u_t+1) = x_t+1
        fx_i = jax.vmap(f)(x_i[:-1], u_sequence) # new_guess. f(x_T) doesn't exist, since x_T is the final state
        fx_i = jnp.concat([initial_state[None, :], fx_i]) # prepend the initial state again -> new x_0,...,x_T
        if (m:=merit_function(x_i[1:], fx_i[1:])) < eps: # if the approximation is good enough, we stop
            print(f"Reached merit={m:.3g} in {step} iterations")
            break
        if step > max_steps:
            print(f"Exiting with merit={m:.3g}, max_tries reached.")
            break
        x_i = fx_i # old_guess <- new_guess
    return fx_i[1:]


def parallel_picard(step_fn, u_sequence, initial_state, eps=1e-15, max_steps=100, initial_traj=None, *, key):

    if initial_traj is None:
        key, x_i = initial_guess(u_sequence, initial_state, key)
    else: x_i = initial_traj

    f = lambda x, u : step_fn(u, x)[0] # f(x_t, u_t+1) = x_t+1 
    step=0
    while True:
        step+=1

        fx_i = jax.vmap(f)(x_i[:-1], u_sequence) # new_guess. f(x_T) doesn't exist, since x_T is the final state

        g_i = fx_i - x_i[:-1] # convert f to residual form g = f - identity
        b_sequence = g_i   # apply damping to the residual
        b_scan = jax.lax.associative_scan(operator_add, b_sequence)
        x_i_p1 = jax.vmap(lambda b: initial_state + b)(b_scan)

        if (m:=merit_function(x_i[1:], x_i_p1)) < eps: # if the approximation is good enough, we stop
            print(f"Reached merit={m:.3g} in {step} iterations")
            break
        if step > max_steps:
            print(f"Exiting with merit={m:.3g}, max_tries reached.")
            break

        x_i = jnp.concat([initial_state[None], x_i_p1]) # old_guess <- new_guess

    return x_i_p1 


def sequential_picard(step_fn, u_sequence, initial_state, eps=1e-15, max_steps=100, initial_traj=None, delta:float=1., *, key):
    if initial_traj is None:
        key, x_i = initial_guess(u_sequence, initial_state, key)
        x_i = x_i * 0
    else: 
        x_i = initial_traj
    
    f = lambda x, u: step_fn(u, x)[0]  # f(x_t, u_t) = x_{t+1}
    step = 0
    
    while True:
        step += 1
        
        # Compute f(x^i_t, u_t) for all t
        fx_i = jax.vmap(f)(x_i[:-1], u_sequence)
        
        # Compute residuals: g(x^i_t) = f(x^i_t) - x^i_t
        g_i = fx_i - x_i[:-1]
        
        # Apply damping
        b_sequence = g_i
        
        # Compute cumulative sum: x^{i+1}_t = x_0 + sum_{s=0}^{t-1} b_s
        x_i_p1 = initial_state + jnp.cumsum(b_sequence, axis=0)
        
        # Check convergence
        if (m := merit_function(x_i[1:], x_i_p1)) < eps:
            print(f"Reached merit={m:.3g} in {step} iterations")
            break
        if step > max_steps:
            print(f"Exiting with merit={m:.3g}, max_tries reached.")
            break
        
        # Update for next iteration
        x_i = jnp.concat([initial_state[None], x_i_p1])
    
    return x_i_p1


def parallel_newton(step_fn, u_sequence, initial_state, eps=1e-15, max_steps=100, initial_traj=None, *, key):
    
    if initial_traj is None:
        key, x_i = initial_guess(u_sequence, initial_state, key)
    else: x_i = initial_traj

    f = lambda x, u : step_fn(u, x)[0] # f(x_t, u_t+1) = x_t+1 
    J_f = lambda x, u: jax.jacrev(step_fn, argnums=1)(u, x)[0] # df(x, u)/ dx 

    step = 0

    while True: # iterate guessing the whole sequence, until the guess is good enough
        step += 1

        fx_i = jax.vmap(f)(x_i[:-1], u_sequence) # new_guess. f(x_T) doesn't exist, since x_T is the final state
        Jfx_i = jax.vmap(J_f)(x_i[:-1], u_sequence) # same as above, we don't need the jacobian of x_T

        b_sequence = fx_i - jax.vmap(lambda J, x: J @ x)(Jfx_i, x_i[:-1])
        
        A_scan, b_scan = jax.lax.associative_scan(operator_full, (Jfx_i, b_sequence))
        x_i_p1 = jax.vmap(lambda A, b: A @ initial_state + b)(A_scan, b_scan)
        
        if (m:=merit_function(x_i[1:], x_i_p1)) < eps: # if the approximation is good enough, we stop
            print(f"Reached merit={m:.3g} in {step} iterations")
            break
        if step > max_steps:
            print(f"Exiting with merit={m:.3g}, max_tries reached.")
            break
        x_i = jnp.concat([initial_state[None], x_i_p1]) # old_guess <- new_guess
    return x_i_p1


def parallel_qnewton(step_fn, u_sequence, initial_state, eps=1e-15, max_steps=100, initial_traj=None, *, key):

    if initial_traj is None:
        keys = jr.split(key, len(initial_state))

        key, x_i = jax.tree.map(initial_guess, (u_sequence for _ in range(len(initial_state))), initial_state, keys)
    else: x_i = initial_traj

    f = lambda x, u : step_fn(u, x)[0] # f(x_t, u_t+1) = x_t+1 
    step=0
    while True:
        step+=1

        fx_i = jax.vmap(f)(x_i[:-1], u_sequence) # new_guess. f(x_T) doesn't exist, since x_T is the final state
        dJfx_i = jax.vmap(diag_jacobian, in_axes=(None, 0))(f, (x_i[:-1], u_sequence))

        b_sequence = fx_i - jax.vmap(lambda dJ, x: dJ * x)(dJfx_i, x_i[:-1])

        A_scan, b_scan = jax.lax.associative_scan(operator_diag, (dJfx_i, b_sequence))
        x_i_p1 = jax.vmap(lambda A, b: A * initial_state + b)(A_scan, b_scan)

        if (m:=merit_function(x_i[1:], x_i_p1)) < eps: # if the approximation is good enough, we stop
            print(f"Reached merit={m:.3g} in {step} iterations")
            break
        if step > max_steps:
            print(f"Exiting with merit={m:.3g}, max_tries reached.")
            break

        x_i = jnp.concat([initial_state[None], x_i_p1]) # old_guess <- new_guess

    return x_i_p1
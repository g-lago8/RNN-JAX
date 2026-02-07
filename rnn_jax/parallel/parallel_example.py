import jax
from rnn_jax.parallel.parallelize import parallel_qnewton
from rnn_jax.cells import LongShortTermMemoryCell

key = jax.random.key(0)
u_sequence = jax.random.normal(key, (50, 10))  # sequence of 50 inputs of dimension 10
lstm = LongShortTermMemoryCell(idim=10, hdim=20, key=key)

parallel_qnewton(step_fn=lstm, 
                u_sequence=u_sequence, 
                initial_state=(
                    jax.numpy.zeros((20,)), 
                    jax.numpy.zeros((20,))
                ), 
                eps=1e-6, 
                max_steps=50, 
                key=key)


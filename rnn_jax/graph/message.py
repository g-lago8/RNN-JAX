import equinox as eqx

class MessageFunction(eqx.Module):
    def __call__(self, m):
        pass

class LinearMessage(MessageFunction):
    linear: eqx.nn.Linear
    def __init__(self, mdim, *, key):
        self.linear = eqx.nn.Linear(mdim, mdim, key=key)
        


    
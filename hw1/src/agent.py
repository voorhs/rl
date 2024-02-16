class Agent:
    def update(self, elite_states, elite_actions):
        raise NotImplementedError()
    
    def predict(self, states):
        raise NotImplementedError()

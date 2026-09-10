

class BaseAgent(object):
    def __init__(self, args):
        self.args = args
        self.nn_models = None
    
    def load_state_dict(self, state_dict):
        for key, model in self.nn_models.items():  
            model.to(device="cpu")
            model.load_state_dict(state_dict[key])
            model.to(device=self.args.device)
    
    def get_state_dict(self):
        state_dict = {}
        for key, model in self.nn_models.items():
            state_dict[key] = model.state_dict()
        return state_dict
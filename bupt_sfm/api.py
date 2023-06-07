from multiprocessing import Value
class API:
    def __init__(self):
        self.shoot_once = Value("b", False)
        self.stop_shoot = Value("b", False)
        self.config_dict = None

api = API()
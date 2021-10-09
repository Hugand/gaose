
class WeightChangeFunction:
    def __init__(self):
        return
    
    @staticmethod
    def quadratic_weight_change(error, learning_rate):
        return (error ** (1 / learning_rate)) * learning_rate
    
    @staticmethod
    def linear_weight_change(error, learning_rate):
        return error * learning_rate
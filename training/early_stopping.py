"""
Early Stopping implementation for training loop.

Monitors a metric (validation accuracy) and stops training when it stops improving.

Arguments:
    patience (int): Number of epochs to wait for improvement before stopping. Default: 5
    min_delta (float): Minimum change to qualify as improvement. Default: 0.001
    mode (str): 'max' for metrics where higher is better (accuracy), 'min' for metrics where lower is better (loss)
"""
    


class EarlyStopping:
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_value = None
        self.should_stop = False
        
        if mode not in ['max', 'min']:
            raise ValueError("mode must be 'max' or 'min'")
    
    def check(self, current_value: float) -> bool:
        
        # First epoch - initialize best_value
        if self.best_value is None:
            self.best_value = current_value
            return False
        
        # Check if there's improvement
        if self.mode == 'max':
            improved = current_value > (self.best_value + self.min_delta)
        else:  # mode == 'min'
            improved = current_value < (self.best_value - self.min_delta)
        
        if improved:
            # Improvement found - reset counter and update best value
            self.best_value = current_value
            self.counter = 0
        else:
            # No improvement - increment counter
            self.counter += 1
            
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_value = None
        self.should_stop = False

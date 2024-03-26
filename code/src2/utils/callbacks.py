class EarlyStoppingCallback:
    """
    Early stopping callback
    """

    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best = float('inf')

    def call(self, loss):
        """
        Call the early stopping callback

        Args:
            loss (float): loss value

        Returns:
            bool: whether to stop or not
        """
        if loss < self.best - self.min_delta:
            self.best = loss
            self.wait = 0
            return False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True
            return False
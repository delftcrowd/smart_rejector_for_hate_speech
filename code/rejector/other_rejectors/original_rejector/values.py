class Values():
    """Contains information about the values
    """

    def __init__(self, value_correct: float, value_incorrect: float, value_rejection: float):
        """
        Args:
            value_correct (float): value of correct prediction
            value_incorrect (float): value of incorrect prediction
            value_rejection (float): value of rejection
        """
        self.value_correct = value_correct
        self.value_incorrect = value_incorrect
        self.value_rejection = value_rejection

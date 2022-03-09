class Costs():
    """Contains information about the costs
    """

    def __init__(self, cost_correct: float, cost_incorrect: float, cost_rejection: float):
        """
        Args:
            cost_correct (float): cost of correct prediction
            cost_incorrect (float): cost of incorrect prediction
            cost_rejection (float): cost of rejection
        """
        self.cost_correct = cost_correct
        self.cost_incorrect = cost_incorrect
        self.cost_rejection = cost_rejection

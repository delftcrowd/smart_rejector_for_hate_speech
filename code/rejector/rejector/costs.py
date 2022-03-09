class Costs():
    """Contains information about the costs
    """

    def __init__(self, cost_TP: float, cost_TN: float, cost_FP: float, cost_FN: float, cost_rejection: float):
        """
        Args:
            cost_TP (float): cost of True Positive
            cost_TN (float): cost of True Negative
            cost_FP (float): cost of False Positive
            cost_FN (float): cost of False Negative
            cost_rejection (float): cost of rejection
        """
        self.cost_TP = cost_TP
        self.cost_TN = cost_TN
        self.cost_FP = cost_FP
        self.cost_FN = cost_FN
        self.cost_rejection = cost_rejection

class Values:
    """Contains information about the values"""

    def __init__(
        self,
        value_TP: float,
        value_TN: float,
        value_FP: float,
        value_FN: float,
        value_rejection: float,
    ):
        """
        Args:
            value_TP (float): value of True Positive
            value_TN (float): value of True Negative
            value_FP (float): value of False Positive
            value_FN (float): value of False Negative
            value_rejection (float): value of rejection
        """
        self.value_TP = value_TP
        self.value_TN = value_TN
        self.value_FP = value_FP
        self.value_FN = value_FN
        self.value_rejection = value_rejection

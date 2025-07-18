class Metrics:
    def __init__(self, accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0, fpr=0.0, tpr=0.0, mcc=0.0, kappa=0.0):
        """
        Encapsulates performance metrics for a classification task.

        Args:
            accuracy (float): Accuracy of the classification.
            precision (float): Precision of the classification.
            recall (float): Recall of the classification.
            f1_score (float): F1-score of the classification.
            fpr (float): False Positive Rate.
            tpr (float): True Positive Rate.
            mcc (float): Matthews Correlation Coefficient.
            kappa (float): Cohen's Kappa Coefficient.
        """
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.fpr = fpr
        self.tpr = tpr
        self.mcc = mcc
        self.kappa = kappa

    def __repr__(self):
        return (f"Metrics(accuracy={self.accuracy:.2f}, precision={self.precision:.2f}, recall={self.recall:.2f}, "
                f"f1_score={self.f1_score:.2f}, fpr={self.fpr:.2f}, tpr={self.tpr:.2f}, mcc={self.mcc:.2f}, kappa={self.kappa:.2f})")
    
    def to_dict(self):
        """
        Converts the Metrics object to a dictionary.

        Returns:
            dict: Dictionary representation of the Metrics object.
        """
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "fpr": self.fpr,
            "tpr": self.tpr,
            "mcc": self.mcc,
            "kappa": self.kappa
        }
    
    def csv_header(self):
        """
        Returns the CSV header string.

        Returns:
            str: A comma-separated string of the metric names.
        """
        return "accuracy,precision,recall,f1_score,fpr,tpr,mcc,kappa"

    def to_csv(self):
        """
        Converts the Metrics object to a CSV row string.

        Returns:
            str: A comma-separated string of the metric values.
        """
        return f"{self.accuracy:.3f},{self.precision:.3f},{self.recall:.3f},{self.f1_score:.3f},{self.fpr:.3f},{self.tpr:.3f},{self.mcc:.3f},{self.kappa:.3f}"
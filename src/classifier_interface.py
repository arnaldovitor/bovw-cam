from abc import ABC, abstractmethod
from typing import Any


class ClassifierInterface(ABC):
    def __init__(self, model: Any) -> None:
        """Initialize the ClassifierInterface.

        Args:
            model (Any): The classifier model.
        """
        self.model = model

    @abstractmethod
    def predict(self, input_sample: Any) -> int:
        """Predict the class label for the input sample.

        Args:
            input_sample (Any): The input sample to classify.

        Returns:
            int: The predicted class label.
        """
        pass

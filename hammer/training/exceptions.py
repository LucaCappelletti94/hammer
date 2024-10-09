"""Custom exceptions for the training module."""


class UnknownPathwayNameError(ValueError):
    """Error raised when an unknown pathway name is encountered."""

    def __init__(self, pathway_name: str, superclass_name: str):
        """Initialize the error."""
        super().__init__(
            f"Unknown pathway name: {pathway_name}, identified from superclass: {superclass_name}"
        )


class UnknownSuperclassNameError(ValueError):
    """Error raised when an unknown superclass name is encountered."""

    def __init__(self, superclass_name: str, class_name: str):
        """Initialize the error."""
        super().__init__(
            f"Unknown superclass name: {superclass_name}, identified from class: {class_name}"
        )

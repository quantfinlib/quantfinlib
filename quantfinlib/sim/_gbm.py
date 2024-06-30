"""This is the gmb module documentation
"""

__all__ = ["GeometricBrownianMotion"]


class GeometricBrownianMotion:
    r"""GeometricBrownianMotion class

    It simulate gbm
    """

    def __init__(self, a: int, b: int):
        r"""Constructor

        Parameters
        ----------
        a
            a number
        b
            another number
        """
        self.a = a
        self.b = b

    def member(self, p: int) -> float:
        r"""A member function

        Parameters
        ----------
        P
            the p  number

        Returns
        -------
        float
            The product of everything

        """
        return self.a * self.b * p

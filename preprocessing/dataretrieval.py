from __future__ import annotations

import glob
import os
from abc import ABC, abstractmethod
from typing import List


class Dataretrieval():
    """
    The Context defines the interface of interest to clients.
    TASK: Retrieve data from arbitrary sources and clean for our use case
    INPUT: Strategie wie zum Beispiel Chalearn bereinigen oder SIgnDict APi nutzen...
    OUTPUT: Data in unit format (which words and saved as MP4) for further development
    """

    def __init__(self, strategy: Strategy) -> None:
        """
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """

        self._strategy = strategy

    @property
    def strategy(self) -> Strategy:
        """
        The Context maintains a reference to one of the Strategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """

        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """

        self._strategy = strategy

    def retrieve(self) -> None:
        """
        The Context delegates some work to the Strategy object instead of
        implementing multiple versions of the algorithm on its own.
        """

        # ...

        print("Context: Sorting data using the strategy (not sure how it'll do it)")
        self._strategy.do_algorithm()
        #print(",".join(result))

        # ...


class Strategy(ABC):
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    @abstractmethod
    def do_algorithm(self):
        pass


"""
Concrete Strategies implement the algorithm while following the base Strategy
interface. The interface makes them interchangeable in the Context.
"""


class CleanCHALearn(Strategy):
    """
    CleanCHALEARN
    TASK: deletes depth images
    """

    def do_algorithm(self):
        for f in glob.glob("../data/train/signer*_depth.mp4"):
            os.remove(f)
        pass


class RetrieveSIGNDICT(Strategy):
    # TODO
    def do_algorithm(self):
        pass

from __future__ import annotations

import glob
import os
import shutil
from abc import ABC, abstractmethod
from typing import Union, Optional

import pandas as pd
from loguru import logger
import csv


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

    def retrieve(self, word_list, source_folder: Optional[str] = None,
                 destination_folder: Optional[str] = None, source_CSV_path: Optional[str] = None,
                 dest_CSV_path: Optional[str] = None) -> None:
        """
        The Context delegates some work to the Strategy object instead of
        implementing multiple versions of the algorithm on its own.
        """

        # ...

        print("Context: Sorting data using the strategy (not sure how it'll do it)")
        self._strategy.do_algorithm(word_list, source_folder, destination_folder, source_CSV_path, dest_CSV_path)
        # print(",".join(result))

        # ...


class Strategy(ABC):
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    @abstractmethod
    def do_algorithm(self, word_numbers_to_filter: list, source_folder: Optional[str] = None,
                     destination_folder: Optional[str] = None):
        pass


"""
Concrete Strategies implement the algorithm while following the base Strategy
interface. The interface makes them interchangeable in the Context.
"""


class CleanCHALearn(Strategy):
    """
    CleanCHALEARN
    TASK: deletes depth images, filter out spefic words, to have weniger klassen
    """

    def do_algorithm(self, word_numbers_to_filter: list[Union[int, str]], source_folder: Optional[str] = None,
                     destination_folder: Optional[str] = None, source_CSV_path: Optional[str] = None,
                     dest_CSV_path: Optional[str] = None):
        if not source_folder:
            source_folder = "data/raw/"

        if not destination_folder:
            destination_folder = "data/train/"

        for f in glob.glob("data/train/signer*_depth.mp4"):
            os.remove(f)

        if isinstance(word_numbers_to_filter[0], str):
            print(word_numbers_to_filter)
            dictionary = pd.read_csv('data/testdata/SignList_ClassId_TR_EN.csv').set_index('EN').to_dict()['ClassId']
            word_numbers_to_filter = [dictionary[word] for word in word_numbers_to_filter]

        df = pd.read_csv(source_CSV_path, header=None)
        df = df.loc[df[1].isin(word_numbers_to_filter)]

        df = df.reset_index()
        with open(dest_CSV_path, 'w', newline='') as f:
            thewriter = csv.writer(f)
            thewriter.writerow(['file_name', 'word'])
            for index, row in df.iterrows():
                thewriter.writerow([row[0] + '_skeleton.jpg', row[1]])
        # move videos to dest
        for example in df[0]:
            if example + "_color.mp4" in os.listdir(source_folder):
                shutil.copy(source_folder + example + "_color.mp4", destination_folder + example + ".mp4")
                logger.info(
                    'Moved ' + example + " from " + source_folder + example + " to " + destination_folder + example)
        pass


class RetrieveSIGNDICT(Strategy):
    # TODO
    def do_algorithm(self):
        pass

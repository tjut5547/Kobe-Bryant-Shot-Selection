
import os, sys
import pandas as pd

class split(object):
    def __init__(self, input_file):
        self._all_dataFrame = ps.read_csv(input_file)
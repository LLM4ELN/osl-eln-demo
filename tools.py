import os
from pathlib import Path
import pandas as pd

class pathIterator:
    def __init__(self, path):
        self.path = path
        self.prompt = (
            'Convert these text lines either key-value pairs? If a unit is present make it such that "Temperature \[C]": 23. '
            'If the data is a tabular, return the table data in json format. '
            'Give your answer as json.dumps')
        self._restart()

    def _restart(self):
        self.files = []
        for path, dirs, files in os.walk(self.path):
            for file in files:
                self.files.append(Path(path) / file)



    def __iter__(self):
        """
        Python iterator

        Returns:
        indentation: iterator
        """
        self._restart()
        return self


    def __next__(self):
        """
        Go to next iterator

        Returns:
        str: test name
        """
        if self.files:
            file = self.files.pop()
            if file.suffix == '.xlsx':
                return file, xls2txt(file)
            if file.suffix == '.csv':
                return file, ''
            if file.suffix == '.pdf':
                return file, ''
            return None, None
        raise StopIteration



    def get_files_by_suffix(self, suffix):
        return [f for f in self.files if f.suffix == suffix]


def xls2txt(filePath):
    res = ''
    df = pd.read_excel(filePath, header=None)
    for _, row in df.iterrows():
        non_null = row.dropna()
        res += ', '.join(str(i) for i in non_null.tolist())+'\n'
    return res


import os
from pathlib import Path
import pandas as pd

class pathIterator:
    def __init__(self, path):
        self.path = path
        self.prompt = (
            'Convert these text lines either key-value pairs? If a unit is present make it such that "Temperature [C]": 23. '
            'If the data is a tabular, return the table data in json format. '
            'Generate only valid JSON output. Do not include any natural language, explanations, markdown, or additional text. '
            'The output must be a single, well-formed JSON object or array. Do not wrap the JSON in code blocks or quotes. \n'
            'Example: {"result": "success", "data": [1, 2, 3]}')
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

    def get(self, namePart):
        """ Go through list and return answer for this file
        """
        for file in self.files:
            if namePart in file.name:
                if file.suffix == '.xlsx':
                    return file, xls2txt(file)
                if file.suffix == '.csv':
                    return file, ''
                if file.suffix == '.pdf':
                    return file, ''
        return '', ''




def xls2txt(filePath):
    res = ''
    df = pd.read_excel(filePath, header=None)
    for _, row in df.iterrows():
        non_null = row.dropna()
        res += ', '.join(str(i) for i in non_null.tolist())+'\n'
    return res

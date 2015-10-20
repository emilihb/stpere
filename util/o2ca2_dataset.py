import numpy as np
from itertools import islice


class O2CA2Dataset:
    def __init__(self, filename, num_registers=None, comments="%", delimiter=" "):
        """ Loads files in a dictionary with a sensor id

        :param filename: filenames of the dataset
        :param num_registers: max lines per dataset file including comments
        :param comments: character for comments
        :param delimiter: character for delimiter
        :type filename: dictionary of form {'id':filename}
        :type num_registers: integer > 0
        :type comments: character
        :type delimiter: character
        """
        self._data = {}

        print num_registers
        for k, v in filename.iteritems():
            with open(v) as f:
                self._data[k] = np.genfromtxt(
                    islice(f, num_registers),
                    comments=comments,
                    delimiter=delimiter)

        self._index = {k: 0 for k in self._data}
        self._timestamp = {k: self._data[k][v][0] for k, v in self._index.iteritems()}

    def next(self):
        """ Returns next sensor type and data entry according to timestamp.
        Raises EOFError when one or more components are not present

        :return: sensor type and data register
        :rtype: 2-element tuple (type, data)
        """
        try:
            _type = min(self._timestamp,  key=self._timestamp.get)
            data = self._data[_type][self._index[_type]]
            self._index[_type] += 1
            self._timestamp[_type] = self._data[_type][self._index[_type]][0]  # timestamp as a value
            return (_type, data)
        except Exception as e:
            raise EOFError(e)


def main():
    fn = {"dvl": "../experiment3/_040825_1735_DVL.log"}
    dset = O2CA2Dataset(fn, 20)
    try:
        while True:
            t, d = dset.next()
            print t, d
    except EOFError:
        print "DONE"

if __name__ == "__main__":
    main()

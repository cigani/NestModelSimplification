import abc


class AbstractSimulation:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        return

    @abs.abstractmethod
    def constructNetwork(self):
        return



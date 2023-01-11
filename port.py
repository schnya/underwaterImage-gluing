from abc import ABCMeta, abstractmethod


class Port(metaclass=ABCMeta):
    @abstractmethod
    def measureDistanceTraveledByKeyPoint(self, matches, q_kp, t_kp, K=5) -> tuple[int, int]:
        raise NotImplementedError()

    @abstractmethod
    def createEmptyImage(self, base_height, base_width, overlay_height, overlay_width, x, y):
        raise NotImplementedError()

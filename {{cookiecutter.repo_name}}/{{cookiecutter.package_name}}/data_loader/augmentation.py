import abc

import torchvision.transforms as T


class AugmentationFactoryBase(abc.ABC):
    def build_transforms(self, train):
        return self.build_train() if train else self.build_test()

    @abc.abstractmethod
    def build_train(self):
        pass

    @abc.abstractmethod
    def build_test(self):
        pass


class MNISTTransforms(AugmentationFactoryBase):

    MEANS = [0]
    STDS = [1]

    def build_train(self):
        return T.Compose([T.ToTensor(), T.Normalize(self.MEANS, self.STDS)])

    def build_test(self):
        return T.Compose([T.ToTensor(), T.Normalize(self.MEANS, self.STDS)])

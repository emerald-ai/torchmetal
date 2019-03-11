import torchvision.transforms.functional as F

class Rotation(object):
    def __init__(self, angle, resample=False, expand=False, center=None):
        super(Rotation, self).__init__()
        if isinstance(angle, (list, tuple)):
            self._angles = angle
            self.angle = None
        else:
            if angle % 360 == 0:
                import warnings
                warnings.warn('Applying a rotation of {0} degrees as a class '
                    'augmentation on a dataset is equivalent to the original '
                    'dataset. Ignoring this class augmentation.'.format(angle),
                    UserWarning, stacklevel=2)
            self._angles = [angle]
            self.angle = angle

        self.resample = resample
        self.expand = expand
        self.center = center

    def __iter__(self):
        return iter(Rotation(angle, resample=self.resample, expand=self.expand,
            center=self.center) for angle in self._angles)

    def __call__(self, image):
        if self.angle is None:
            raise ValueError()
        return F.rotate(image, self.angle, self.resample,
                        self.expand, self.center)

    def __repr__(self):
        if self.angle is None:
            return 'Rotation({0})'.format(', '.join(map(str, self._angles)))
        else:
            return 'Rotation({0})'.format(self.angle % 360)

class HorizontalFlip(object):
    def __iter__(self):
        return iter([HorizontalFlip()])

    def __call__(self, image):
        return F.hflip(image)

    def __repr__(self):
        return 'HorizontalFlip()'

class VerticalFlip(object):
    def __iter__(self):
        return iter([VerticalFlip()])

    def __call__(self, image):
        return F.vflip(image)

    def __repr__(self):
        return 'VerticalFlip()'

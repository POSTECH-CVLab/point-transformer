import numpy as np

import torch


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, points, color):
        for t in self.transforms:
            points, color= t(points, color)
        return points, color
    
    def __repr__(self):
        return 'Compose(\n' + '\n'.join(['\t' + t.__repr__() + ',' for t in self.transforms]) + '\n)'
            

class ToTensor(object):
    def __call__(self, data, label):
        data = torch.from_numpy(data)
        if not isinstance(data, torch.FloatTensor):
            data = data.float()
        label = torch.from_numpy(label)
        if not isinstance(label, torch.LongTensor):
            label = label.long()
        return data, label


class RandomRotate(object):
    def __init__(self, rotate_angle=None, along_z=True, color_rotate=False):
        self.rotate_angle = rotate_angle
        self.along_z = along_z
        self.color_rotate = color_rotate

    def __call__(self, points, color):
        if self.rotate_angle is None:
            rotate_angle = np.random.uniform() * 2 * np.pi
        else:
            rotate_angle = self.rotate_angle
        cosval, sinval = np.cos(rotate_angle), np.sin(rotate_angle)
        if self.along_z:
            rotation_matrix = np.array([[cosval, sinval, 0], [-sinval, cosval, 0], [0, 0, 1]])
        else:
            rotation_matrix = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]])
        points[:, 0:3] = np.dot(points[:, 0:3], rotation_matrix)
        if self.color_rotate:
            color[:, 0:3] = np.dot(color[:, 0:3], rotation_matrix)
        return points, color
    
    def __repr__(self):
        return 'RandomRotate(rotate_angle: {}, along_z: {})'.format(self.rotate_angle, self.along_z)


class RandomRotatePerturbation(object):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma = angle_sigma
        self.angle_clip = angle_clip

    def __call__(self, data, label):
        angles = np.clip(self.angle_sigma*np.random.randn(3), -self.angle_clip, self.angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        data[:, 0:3] = np.dot(data[:, 0:3], R)
        if data.shape[1] > 3:  # use normal
            data[:, 3:6] = np.dot(data[:, 3:6], R)
        return data, label


class RandomScale(object):
    def __init__(self, scale_low=0.8, scale_high=1.2):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, points, color):
        scale = np.random.uniform(self.scale_low, self.scale_high)
        points[:, 0:3] *= scale
        return points, color

    def __repr__(self):
        return 'RandomScale(scale_low: {}, scale_high: {})'.format(self.scale_low, self.scale_high)


class RandomShift(object):
    def __init__(self, shift_range=0.1):
        self.shift_range = shift_range

    def __call__(self, points, color):
        shift = np.random.uniform(-self.shift_range, self.shift_range, 3)
        points[:, 0:3] += shift
        return points, color

    def __repr__(self):
        return 'RandomShift(shift_range: {})'.format(self.shift_range)


class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, points, color):
        assert (self.clip > 0)
        jitter = np.clip(self.sigma * np.random.randn(points.shape[0], 3), -1 * self.clip, self.clip)
        points[:, 0:3] += jitter
        return points, color
    
    def __repr__(self):
        return 'RandomJitter(sigma: {}, clip: {})'.format(self.sigma, self.clip)


class ChromaticAutoContrast(object):
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, points, color):
        if np.random.rand() < self.p:
            lo = np.min(color, axis=0, keepdims=True)
            hi = np.max(color, axis=0, keepdims=True)
            scale = 255 / (hi - lo)
            contrast_color = (color - lo) * scale
            blend_factor = np.random.rand() if self.blend_factor is None else self.blend_factor
            color = (1 - blend_factor) * color + blend_factor * contrast_color
        return points, color


class ChromaticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, points, color):
        if np.random.rand() < self.p:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            color = np.clip(tr + color, 0, 255)
        return points, color


class ChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005):
        self.p = p
        self.std = std

    def __call__(self, points, color):
        if np.random.rand() < self.p:
            noise = np.random.randn(color.shape[0], 3)
            noise *= self.std * 255
            color[:, :3] = np.clip(noise + color[:, :3], 0, 255)
        return points, color


class HueSaturationTranslation(object):
    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def __init__(self, hue_max=0.5, saturation_max=0.2):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, points, color):
        # Assume color[:, :3] is rgb
        hsv = HueSaturationTranslation.rgb_to_hsv(color[:, :3])
        hue_val = (np.random.rand() - 0.5) * 2 * self.hue_max
        sat_ratio = 1 + (np.random.rand() - 0.5) * 2 * self.saturation_max
        hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
        hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
        color[:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)
        return points, color


class RandomDropColor(object):
    def __init__(self, p=0.8, color_augment=0.0):
        self.p = p
        self.color_augment = color_augment
    
    def __call__(self, points, color):
        if color is not None and np.random.rand() > self.p:
            color *= self.color_augment
        return points, color

    def __repr__(self):
        return 'RandomDropColor(color_augment: {}, p: {})'.format(self.color_augment, self.p)
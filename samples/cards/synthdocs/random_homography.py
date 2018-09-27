import numpy as np
import random
import math

class Random_Homography(object):
    def __init__(self, **kwargs):
        print('Using modified synthdocs.random_homography...')
        self.scale_amount = kwargs.get('scale_amount')
        self.rotation_amount = kwargs.get('rotation_amount')
        self.warp_amount = kwargs.get('warp_amount')
        self.translate_amount = kwargs.get('translate_amount')

    def random_homography(self, canvas_size):
        # assume that the rectangle images have been rescaled to the fit the canvas
        final_transform = np.eye(3, dtype=np.float32)

        # translate the cornerpoints such that the centre of the rectangle is at the origin
        tx = canvas_size[0] / 2.
        ty = canvas_size[1] / 2.
        translation_matrix = np.array([[1, 0, -tx],
                                       [0, 1, -ty],
                                       [0, 0, 1]], dtype=np.float32)
        final_transform = np.dot(translation_matrix, final_transform)

        # apply a random rescale (to produce image samples with rectangles of varying sizes)
        # that preserves aspect ratio
        scale_factor = random.uniform(self.scale_amount[0], self.scale_amount[1])
        zoom_matrix = np.array([[scale_factor, 0., 0.],
                                [0., scale_factor, 0.],
                                [0., 0., 1.]], dtype=np.float32)
        final_transform = np.dot(zoom_matrix, final_transform)

        # perform a random rotation on the rectangle cornerpoints
        theta = np.pi / 180 * np.random.uniform(-self.rotation_amount, self.rotation_amount)


        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]], dtype=np.float32)
        final_transform = np.dot(rotation_matrix, final_transform)

        # generate a small amount of perspective warping
        A = random.uniform(-self.warp_amount, self.warp_amount)
        B = random.uniform(-self.warp_amount, self.warp_amount)
        warp_matrix = np.array([[1, 0., 0.],
                                [0., 1, 0.],
                                [A, B, 1.]], dtype=np.float32)
        final_transform = np.dot(warp_matrix, final_transform)

        # generate a translation
        random_translation = np.random.normal(0, self.translate_amount, size=(2))
        translate_matrix = np.array([[1,0,random_translation[0]],
                                     [0,1,random_translation[1]],
                                     [0,0,1]], dtype=np.float32)
        final_transform = np.dot(translate_matrix, final_transform)

        # translate back to centre of canvas
        translation_matrix2 = np.array([[1, 0, tx],
                                        [0, 1, ty],
                                        [0, 0, 1]], dtype=np.float32)
        final_transform = np.dot(translation_matrix2, final_transform)

        return final_transform

    def max_allowed_rotation(self, H, W, h, w):
        '''
        For a rectangle A (with size W,H) and a smaller rectangle B (with size w,h) both sharing the same centroid,
        calculate the maximum amount of rotation that can be applied to B without its corners exceeding the boundaries
        of A.
        :return: float - angle in radians
        '''
        r = math.sqrt((h / 2.) * (h / 2.) + (w / 2.) * (w / 2.))
        theta = math.atan((h / 2.) / (w / 2.))

        q_vert = (H / 2.) / r
        # handle the case in which the card exceeds vertical limits
        if q_vert < -1 or q_vert > 1:
            amount_vert = 2 * math.pi
        else:
            phi = math.asin(q_vert)
            amount_vert = max(0, phi - theta)

        q_horiz = (W / 2.) / r
        # handle the case in which the card exceeds horizontal limits
        if q_horiz < -1 or q_horiz > 1:
            amount_horiz = 2 * math.pi
        else:
            psi = math.acos(q_horiz)
            amount_horiz = max(0, theta - psi)

        return min(amount_horiz, amount_vert)


    # def max_allowed_translation(self, ):
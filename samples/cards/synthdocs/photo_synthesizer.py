'''

USAGE: $python this_script.py ipod_rectangle.jpg background.jpg

'''
import random

from synthdocs import coordinate_geometry
import cv2
import numpy as np
from PIL import Image

from synthdocs.random_homography import Random_Homography


def showims(img_array_list, label_list=None):
    fig = plt.figure()
    for i, img in enumerate(img_array_list):
        a = fig.add_subplot(1, len(img_array_list), i + 1)
        imgplot = plt.imshow(img)
        if label_list is not None:
            a.set_title(label_list[i])
    plt.show()

class PhotoSynthesizer(object):
    '''
    Class for synthesizing "photos" of rectangles. Simulates perspective distortion, scale, rotation and translation.
    Convention for box/cornerpoints ordering:
            [top-left, top-right, bottom-right, bottom-left]
    i.e.    clockwise from top-left.
    '''
    def __init__(self,
                 hsv_jitter_amout,
                 blur_param):
        print('using modified synthdocs.photo_synthesizer...')
        # self.canvas_size = canvas_size
        # self.scale_amount = scale_amount
        # self.rotation_amount = rotation_amount
        # self.warp_amount = warp_amount
        self.hsv_jitter_amount = hsv_jitter_amout
        self.blur_param = blur_param
        # self.translate_amount = translate_amount

        # self.random_homography = Random_Homography(scale_amount, rotation_amount, warp_amount, translate_amount)

    def colour_jitter(self, cv2_image):
        im = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2HSV)
        # h, s, v = cv2.split(im)
        im = im.astype(np.int32)
        h = im[:,:,0]
        s = im[:,:,1]
        v = im[:,:,2]
        h = np.clip((h + np.random.randint(-self.hsv_jitter_amount[0], self.hsv_jitter_amount[0])), 0, 255 )
        s = np.clip((s + np.random.randint(-self.hsv_jitter_amount[1], self.hsv_jitter_amount[1])), 0, 255 )
        v = np.clip((v + np.random.randint(-self.hsv_jitter_amount[2], self.hsv_jitter_amount[2])), 0, 255 )
        # im = cv2.merge([h, s, v])
        im = np.array([h, s, v]).astype(np.uint8)
        im = im.transpose((1, 2, 0))
        im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
        return im

    def add_white_circle(self, image_cv2):
        h, w, c = image_cv2.shape
        centre_x = random.randint(0, w)
        centre_y = random.randint(0, h)
        circle_radius = random.randint(int(w / 100), int(w / 4))
        # choose an amount to blur the circle (to give it a feathered edge)
        blur_kernelsize = random.randint(int(circle_radius / 2), circle_radius)
        constant = random.uniform(0.4, 0.8)
        # print('centre: ({},{})\ncircle radius: {}\nblur kernel size: {}\naddative constant: {}'.format(centre_x, centre_y, circle_radius, blur_kernelsize, constant))
        mask = np.zeros_like(image_cv2, dtype=np.uint8)
        cv2.circle(mask, (centre_x, centre_y), circle_radius, (255, 255, 255), -1)
        # showims([mask], ['mask'])
        mask = cv2.blur(mask, (blur_kernelsize, blur_kernelsize))
        # showims([mask], ['mask blur'])
        mask = mask.astype(np.float32)
        image_cv2 = image_cv2.astype(np.float32)
        image_cv2 += mask
        image_cv2 = np.clip(image_cv2, 0, 255)
        image_cv2 = image_cv2.astype(np.uint8)
        # print('Did a circle!')
        return mask, image_cv2

    def random_kernelsize(self):
        random_kernelsize = int(random.expovariate(lambd=self.blur_param))
        if random_kernelsize % 2 == 0:
            random_kernelsize+=1
        return random_kernelsize

    def blur(self, cv2_image):
        random_kernelsize = self.random_kernelsize()
        # kernel = np.ones((5, 5), np.float32) / 25
        cv2_image = cv2.GaussianBlur(cv2_image,(random_kernelsize, random_kernelsize), 0)
        return cv2_image

    def randomize_background_image(self, im_background):
        im_background = np.array(im_background)
        h,w,c = im_background.shape
        cornerpoints_canvas = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
        jitter_vector = np.random.randint(-int(h/10), int(h/10),(4,2))
        cornerpoints_warped = np.float32(cornerpoints_canvas + jitter_vector)
        H = cv2.getPerspectiveTransform(cornerpoints_canvas, cornerpoints_warped)
        im_background = cv2.warpPerspective(im_background, np.linalg.inv(H), (w,h), borderMode=cv2.BORDER_REFLECT)
        if random.random() < 0.5:
            im_background = cv2.flip(im_background, 0)
        im_background = Image.fromarray(im_background)
        return im_background

    # def snap(self, img_rectangle, img_background, canvas_size):
    #     '''
    #     :param img_rectangle: cv2 image of rectangle
    #     :param img_background_new: cv2 image of background
    #     :return: cv2 image - composite of rectangle and background with random perspective
    #     '''
    #
    #     # convert images from cv2 arrays into PIL images
    #     img_rectangle = Image.fromarray(img_rectangle)
    #     img_background = Image.fromarray(img_background)
    #
    #     # give the ID card an alpha channel
    #     img_rectangle_new = img_rectangle.convert("RGBA")
    #     img_rectangle_new = img_rectangle_new.resize(canvas_size)
    #
    #     # warp the background image slightly
    #
    #     # compute a random projective transform homography
    #     homography = self.random_homography.random_homography(canvas_size)
    #     # only interested in the 8 degrees of freedom, as a tuple
    #     h = homography.flatten()[:8]
    #     # make sure to normalize all the values such that the final entry H33 = 1
    #     normalizer = homography[2][2]
    #     h /= normalizer
    #     h = tuple(h)
    #
    #     # apply the transform
    #     new_image = img_rectangle_new.transform(canvas_size, Image.PERSPECTIVE, h, Image.BICUBIC)
    #
    #     # jitter the background image slightly
    #     img_background = self.randomize_background_image(img_background)
    #
    #     # rescale the background to fit the canvas
    #     img_background_new = img_background.resize(canvas_size)
    #
    #     # paste projective-transformed rectangle onto background
    #     img_background_new.paste(new_image, (0, 0), new_image)
    #
    #     # convert the PIL image to opencv:
    #     img_background_new = np.array(img_background_new)
    #     img_background_new = cv2.cvtColor(img_background_new, cv2.COLOR_RGB2BGR)
    #
    #     # blur and colour jitter
    #     output_photo = self.colour_jitter(img_background_new)
    #     output_photo = self.blur(output_photo)
    #     # output_photo = self.sharpen(output_photo)
    #
    #     # add white blobs
    #     if random.random() < 0.3:
    #         _, output_photo = self.add_white_circle(output_photo)
    #
    #     # define canonical cornerpoints
    #     cornerpoints_original = np.array([[0, 0],
    #                                       [canvas_size[0], 0],
    #                                       [canvas_size[0], canvas_size[1]],
    #                                       [0, canvas_size[1]]], dtype=np.float32)
    #
    #     # transform canonical cornerpoints to their location on the canvas
    #     cornerpoints_warped = coordinate_geometry._transform_cornerpoints_2D(np.linalg.inv(homography), cornerpoints_original)
    #
    #     return output_photo, homography, cornerpoints_warped


    def snap_this_homography(self, img_rectangle, img_background, canvas_size, homography):
        '''
        :param img_rectangle: cv2 image of rectangle
        :param img_background_new: cv2 image of background
        :param homography: the homography to use
        :return: cv2 image - composite of rectangle and background with random perspective
        '''

        # convert images from cv2 arrays into PIL images
        img_rectangle = Image.fromarray(img_rectangle)
        img_background = Image.fromarray(img_background)

        # give the ID card an alpha channel
        img_rectangle_new = img_rectangle.convert("RGBA")
        img_rectangle_new = img_rectangle_new.resize(canvas_size)

        # warp the background image slightly

        # compute a random projective transform homography
        # homography = self.random_homography.random_homography(canvas_size)
        # only interested in the 8 degrees of freedom, as a tuple
        h = homography.flatten()[:8]
        # make sure to normalize all the values such that the final entry H33 = 1
        normalizer = homography[2][2]
        h /= normalizer
        h = tuple(h)

        # apply the transform
        new_image = img_rectangle_new.transform(canvas_size, Image.PERSPECTIVE, h, Image.BICUBIC)

        # jitter the background image slightly
        img_background = self.randomize_background_image(img_background)

        # rescale the background to fit the canvas
        img_background_new = img_background.resize(canvas_size)

        # paste projective-transformed rectangle onto background
        img_background_new.paste(new_image, (0, 0), new_image)

        # convert the PIL image to opencv:
        img_background_new = np.array(img_background_new)
        img_background_new = cv2.cvtColor(img_background_new, cv2.COLOR_RGB2BGR)

        # blur and colour jitter
        output_photo = self.colour_jitter(img_background_new)
        output_photo = self.blur(output_photo)
        # output_photo = self.sharpen(output_photo)

        # add white blobs
        if random.random() < 0.3:
            _, output_photo = self.add_white_circle(output_photo)

        # define canonical cornerpoints
        cornerpoints_original = np.array([[0, 0],
                                          [canvas_size[0], 0],
                                          [canvas_size[0], canvas_size[1]],
                                          [0, canvas_size[1]]], dtype=np.float32)

        # transform canonical cornerpoints to their location on the canvas
        cornerpoints_warped = coordinate_geometry.transform_cornerpoints_2D(np.linalg.inv(homography), cornerpoints_original)

        return output_photo, homography, cornerpoints_warped

if __name__ == '__main__':
    import sys
    from matplotlib import pyplot as plt

    IMPATH_RECTANGLE = '/home/sal9000/PycharmProjects/SynthDocs/synthdocs/bill.png'
    IMPATH_BACKGROUND = '/home/sal9000/PycharmProjects/SynthDocs/synthdocs/background.jpg'


    # the size of the photo to be generated
    CANVAS_SIZE = (640, 410)

    # the (min, max) scale factors to be applied to cards
    SCALE_AMOUNT = (1, 2)

    # the abs(max) value to be placed in the H31 and H32 entries of the normalized homography matrix
    # i.e. amount of perspective distortion
    WARP_AMOUNT = 0.0005

    # the abs(max) number of degrees a card can be rotated
    ROTATION_AMOUNT = 15

    # amount to jitter HSV components
    HSV_JITTER_AMOUNT = (5, 40, 90)

    # amount to blur the image
    BLUR_PARAM = 0.25


    IMG_RECTANGLE = cv2.imread(IMPATH_RECTANGLE)
    IMG_BACKGROUND = cv2.imread(IMPATH_BACKGROUND)

    photographer = PhotoSynthesizer(
                                    scale_amount=SCALE_AMOUNT,
                                    rotation_amount=ROTATION_AMOUNT,
                                    warp_amount=WARP_AMOUNT,
                                    hsv_jitter_amout=HSV_JITTER_AMOUNT,
                                    blur_param=BLUR_PARAM
    )

    cols = [(225, 0, 0), (0, 225, 0), (0, 0, 225), (225, 225, 0)]

    CANVAS_SIZES = [(640,480), (500, 315), (400, 300)]

    while True:

        # print('War were declared')

        canvas_size = random.choice(CANVAS_SIZES)

        photo, homography, cornerpoints = photographer.snap(IMG_RECTANGLE, IMG_BACKGROUND, canvas_size)

        cornerpoints = cornerpoints.astype(int)
        for j in range(4):
            cv2.circle(photo, tuple(cornerpoints[j]), 5, color=cols[j], thickness=-1)

        photo_small = cv2.resize(photo, (320, 320))
        sfx = canvas_size[0]/320.
        sfy = canvas_size[1]/320.
        cornerpoints_small = np.copy(cornerpoints).astype(np.float32)
        cornerpoints_small[:,0] /= sfx
        cornerpoints_small[:,1] /= sfy

        cornerpoints_small = cornerpoints_small.astype(int)
        for j in range(4):
            cv2.circle(photo_small, tuple(cornerpoints_small[j]), 5, color=cols[j], thickness=-1)

        # show(photo)
        # plt.imshow(cv2.cvtColor(photo_small, cv2.COLOR_BGR2RGB))
        # plt.show()
        showims([photo], [canvas_size])
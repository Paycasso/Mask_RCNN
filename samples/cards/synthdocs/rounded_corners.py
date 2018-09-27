from PIL import Image, ImageDraw

def add_corners(im, rad):
    '''
    Add rounded corners to an image, filling in the cropped regions with alpha transparency.
    :param im: PIL Image - the image to have rounded corners added
    :param rad: Number - radius of the corner rounding
    :return: PIL Image - image with rounded corners
    '''
    circle = Image.new('L', (rad * 2, rad * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, rad * 2, rad * 2), fill=255)
    alpha = Image.new('L', im.size, 255)
    w, h = im.size
    alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
    alpha.paste(circle.crop((0, rad, rad, rad * 2)), (0, h - rad))
    alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
    alpha.paste(circle.crop((rad, rad, rad * 2, rad * 2)), (w - rad, h - rad))
    im.putalpha(alpha)
    return im

def corner_ting(size, rad):
    '''
    Creates a simple white rectangle with rounded edges.
    :param size: Int-Tuple - (width, height)
    :param rad: Number - radius of the corner rounding
    :return: PIL Image - Greyscale image; a white rectangle with rounded corners
    '''
    circle = Image.new('L', (rad * 2, rad * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, rad * 2, rad * 2), fill=255)
    alpha = Image.new('L', size, 255)
    w, h = size
    alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
    alpha.paste(circle.crop((0, rad, rad, rad * 2)), (0, h - rad))
    alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
    alpha.paste(circle.crop((rad, rad, rad * 2, rad * 2)), (w - rad, h - rad))
    # alpha.show('this is ALPHA')
    return alpha

if __name__ == '__main__':

    #######################################   RUN ON A SINGLE IMAGE ##################################################
    # IMG_PATH = '/home/hal9000/Lenna.png'
    # IMG_PATH = '/home/hal9000/Paycasso_Data/_1355_20161123/ARG_I1.jpg'
    #
    #
    # im = Image.open(IMG_PATH)
    # im = add_corners(im, 20)
    # im.save('rounded.png')

    corner_ting((500, 315), 25)

    ###################################### RUN ON ALL IMAGES IN A DIRECTORY ##########################################
    # import os
    #
    # INDIR = '/home/hal9000/Paycasso_Data/_1355_20161123'
    # OUTDIR = '/home/hal9000/Paycasso_Data/All_IDs_Rounded'
    #
    # from utilities.utils import mkdir
    # mkdir(OUTDIR)
    #
    # i = 0
    # for imagefile in os.listdir(INDIR):
    #
    #     imagepath = os.path.join(INDIR, imagefile)
    #     prefix = imagefile.split('.')[0]
    #     savepath = os.path.join(OUTDIR, prefix+'.png')
    #
    #     img = Image.open(imagepath)
    #
    #     img = img.resize((500, 315))
    #
    #     img = add_corners(img, 25)
    #
    #     img.save(savepath)
    #
    #     print('=====Resized and added rounded corners: {} out of {}.====='.format(i, len(os.listdir(INDIR))))
    #
    #     i+=1
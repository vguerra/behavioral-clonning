import cv2
import numpy as np
import matplotlib.pyplot as plt

# Global settings
IMAGE_SHAPE = (160, 320, 3)

# Local settings
BRIGHTNESS_RANGE = 0.25
TRANS_X_RANGE = 200
TRANS_Y_RANGE = 40
TRANS_ANGLE = 0.7

# Image manipulation
def load_img(path):
    """
    Loads an image given img_path and applies
    necesary transformations.
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

def do_flip():
    return np.random.randint(2) == 0

def flip(img, angle):
    """
    Flips an image horizontally, and computes
    its angle accordingly.
    """
    return cv2.flip(img, 1), -1. * angle

def normalize(img):
    """
    Performs normalization on image.
    """
    return img/127.5 - 1.0

def translate(img, angle):
    """
    Shifts an image vertically and horizontally
    by a randaom amount

    New angle is computed accordingly.
    """
    x_translation = (TRANS_X_RANGE * np.random.uniform()) - (TRANS_X_RANGE / 2)
    new_angle = angle + ((x_translation / TRANS_X_RANGE) * 2) * TRANS_ANGLE
    y_translation = (TRANS_Y_RANGE * np.random.uniform()) - (TRANS_Y_RANGE / 2)
    translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
    return cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0])), new_angle

def perturb_brightness(img):
    """
    Randomly modifies the brightness of an image
    """
    tmp = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    brightness = BRIGHTNESS_RANGE + np.random.uniform()
    tmp[:, :, 2] = tmp[:, :, 2] * brightness
    return cv2.cvtColor(tmp, cv2.COLOR_HSV2RGB)

def add_shadow(img):
    """
    Places a shadow (with random dimensions) on top of image.
    """
    Y = IMAGE_SHAPE[1]
    X = IMAGE_SHAPE[0]
    top_y = Y*np.random.uniform()
    top_x = 0
    bot_x = X
    bot_y = Y*np.random.uniform()
    image_hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0 : img.shape[0], 0:img.shape[1]][0]
    Y_m = np.mgrid[0 : img.shape[0], 0:img.shape[1]][1]
    
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)] = 1
    
    random_bright = .25 + .7*np.random.uniform()
    if np.random.randint(2)==1:

        cond1 = shadow_mask==1
        cond0 = shadow_mask==0

        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    

    img = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return img

# helpers
def save_img(img, filename):
    """
    Saves an image to disk using filename given.
    """
    plt.imshow(img)
    plt.savefig(filename, format='png')
    plt.clf()
    return

def show_img(img, title):
    """
    Show an image with given title and waits for user input.
    """
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def plot_angle_dist(angles, title):
    plt.hist(angles, bins=200)
    plt.title(title)
    plt.savefig('angle_dist')
    plt.clf()
    return

def plot_loss(loss, val_loss):
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('MSE Loss')
    plt.ylabel('MSE loss')
    plt.xlabel('epoch')
    plt.legend(['Training set', 'Validation set'], loc='upper right')
    plt.savefig('loss')
    plt.clf()
    return

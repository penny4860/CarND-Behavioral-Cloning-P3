
import matplotlib.pyplot as plt

def plot_images(images, titles, n_rows, n_cols):
    fig, ax = plt.subplots()
    for i, img in enumerate(images):
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(img)
        plt.title(titles[i])
        # plt.axis("off")
        # plt.title("angle : {}".format(i, str_labels[i]))
    # plt.subplots_adjust(left=0, bottom=0, right=1.0, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()

from generator.image_preprocess import Preprocessor
from generator.image_augment import CarAugmentor
img = plt.imread("img.jpg")

img_shear, angle1 = CarAugmentor()._random_shear(img, 0.0, 100)
img_flip, angle2 = CarAugmentor()._random_flip(img_shear, angle1, flipping_prob=1.0)
img_gamma = CarAugmentor()._random_gamma(img_flip)

plot_images([img, img_shear, img_flip, img_gamma],
            ["original\n target angle {}".format(0.0),
             "random sheare\n target angle {:.2f}".format(angle1),
             "random flip\n target angle {:.2f}".format(angle2),
             "random gamma\n target angle {:.2f}".format(angle2)], 1, 4)

img_crop = Preprocessor()._crop(img_gamma, 0.35, 0.1)
img_resize = Preprocessor()._resize(img_crop, (64,64))
plot_images([img_gamma, img_crop, img_resize], ["original", "crop", "resize"], 1, 3)


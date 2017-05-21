#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_images(image_list, col=2, gray=False):
    fig = plt.figure(figsize=(20,20))
    num_images = len(image_list)
    for i in range(num_images):
        image = image_list[i]
        ax = fig.add_subplot(int(num_images/col) + 1 , col, i + 1, xticks=[], yticks=[])
        if gray:
            ax.imshow(image.squeeze(), cmap='gray')
        else:
            ax.imshow(image.squeeze())

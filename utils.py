import os, imageio, scipy.misc
import matplotlib.pyplot as plt


def create_gif(gif_name, img_path, duration=0.3):
    frames = []
    img_names = os.listdir(img_path)
    img_list = [os.path.join(img_path, img_name) for img_name in img_names]
    for img_name in img_list:
        frames.append(imageio.imread(img_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)


def visualize_loss(D_list_float, G_list_float):
    list_epoch = list(range(len(D_list_float)))

    full_path = os.path.join(os.getcwd(), "saved/logging.png")
    plt.figure()
    plt.plot(list_epoch, G_list_float, label="generator", color='g')
    plt.plot(list_epoch, D_list_float, label="discriminator", color='b')
    plt.legend()
    plt.title("DCGAN_Anime")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(full_path)

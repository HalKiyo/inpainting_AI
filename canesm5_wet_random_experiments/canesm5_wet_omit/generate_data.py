import os
import numpy as np
import random
from PIL import Image

def main():
    save_dir = '/docker/home/hasegawa/docker-gpu/reconstructionAI'\
               '/canesm5_wet_random_experiments/canesm5_wet_omit/mask'
    save_npy = '/docker/home/hasegawa/docker-gpu/reconstructionAI'\
               '/canesm5_wet_random_experiments/canesm5_wet_omit/data'\
               '/canesm5_wet_omit_mask.npy'

    image_size = [40,128]
    N = 1000
    images = np.zeros((N, image_size[0], image_size[1]))

    for i in range(N):
        canvas = np.ones((image_size[0],image_size[1])).astype("i")
        ini_x = random.randint(0, image_size[0] - 1)
        ini_y = random.randint(0, image_size[1] - 1)
        mask = random_walk(canvas, ini_x, ini_y, 128 ** 2)
        print("save:", i, np.sum(mask))

        img = Image.fromarray(mask * 255).convert('1')
        img.save(f'{save_dir}/{i}.jpg')

        images[i, :, :] = mask

    np.save(save_npy, images)


def random_walk(canvas, ini_x, ini_y, length):
    action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    x = ini_x
    y = ini_y
    img_size = canvas.shape
    x_list = []
    y_list = []
    for i in range(length):
        r = random.randint(0, len(action_list) - 1)
        x = np.clip(x + action_list[r][0], a_min=0, a_max=img_size[0] - 1)
        y = np.clip(y + action_list[r][1], a_min=0, a_max=img_size[1] - 1)
        x_list.append(x)
        y_list.append(y)
    canvas[np.array(x_list), np.array(y_list)] = 0
    return canvas


if __name__ == '__main__':
    main()


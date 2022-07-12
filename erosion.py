import numpy as np
from PIL import Image


def rgbTogray(rgb: np.array) -> np.array:
    # standard graysacale conversion
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def binaryzation(gray: np.array) -> np.array:
    # map like conditional conversion
    # limiarization with threshold at the half
    return (127 < gray) & (gray <= 255)


def erosion(image: np.array, kernel: np.array) -> np.array:
    blank_image = np.zeros_like(image)
    output: np.array = blank_image

    image_padded = np.zeros(
        (image.shape[0] + kernel.shape[0] - 1,
         image.shape[1] + kernel.shape[1] - 1)
    )

    # Copy image to padded image
    image_padded[kernel.shape[0] - 2: -1, kernel.shape[1] - 2: -1] = image

    # Iterate over image & apply kernel
    for y in range(image.shape[1]):
        for x in range(image.shape[0]):
            summation = (
                kernel * image_padded[x: x +
                                      kernel.shape[0], y: y + kernel.shape[1]]
            )

            output[x, y] = int(np.count_nonzero(summation)
                               == np.count_nonzero(kernel))
    return output


# kernel to be applied
structuring_element = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

if __name__ == "__main__":
    # read original image
    image = Image.open(r"./assets/example.jpg")

    filename = image.filename.split('/')[-1]

    image = np.array(image)

    dimension = len(image.shape)

    image = rgbTogray(image) if dimension == 3 else image

    image = binaryzation(image)

    # # Apply erosion operation to a binary image
    output = erosion(image, structuring_element)
    # # Save the output image
    pil_img = Image.fromarray(output)
    pil_img.save(r"./outputs/eroded{}".format(filename))

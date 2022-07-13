from PIL import Image
import numpy as np
from erosion import erode


def main():
    structuring_element = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    # read original image
    image = Image.open(r"./assets/lena.jpg")

    filename = image.filename.split('/')[-1]

    image = np.array(image)

    # # Apply erosion operation to a binary image
    output = erode(image, structuring_element, backToRgb=True)

    # # Save the output image
    pil_img = Image.fromarray(output)
    pil_img.save(r"./output/eroded_{}".format(filename))


    # kernel to be applied
if __name__ == "__main__":
    main()

from PIL import Image
from network import Network
import numpy as np
import mnist_loader


def my():
    (training_data, validation_data, test_data) = mnist_loader.load_data_wrapper()
    example_data = (255-255*training_data[101][0]).astype(np.uint8).reshape(28,28)
    example_img = Image.fromarray(example_data, "L")
    example_img.show()
    a = 1
    print("hello: {}".format(a))
    a += 1
    print("hello: {}".format(a))
    nw = Network([28 * 28, 16, 16, 10])
    x = np.random.randn(nw.sizes[0], 1)
    a = nw.feedforward(x)

    b = 2


if __name__ == "__main__":
    my()

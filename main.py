from data.load_data import mnist_data
from models.unet import build_unet
from training import train, sample

def main():
    x_train, _, _, _ = mnist_data()
    model = build_unet()
    epochs = 10
    train.train(model, x_train, epochs=epochs)


if __name__ == "__main__":
    main()
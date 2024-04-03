import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import make_grid
from model import Generator, Discriminator, weight_init
from AnimeDataset import AnimeDataset
import numpy as np
import imageio
import os, argparse
from tqdm import tqdm
from utils import create_gif, visualize_loss
from torchvision.utils import save_image
def main():
    parser = argparse.ArgumentParser(description="Train a GAN on a dataset of anime faces")

    parser.add_argument("--learning_rate", type=float, default=0.0001,
                       help="Learning rate for both the generator and discriminator")
    parser.add_argument("--beta1", type=float, default=0.5,
                       help="Beta1 hyperparameter for the Adam optimizer")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Number of samples per batch")
    parser.add_argument("--dataset_path", type=str, default="faces",
                       help="Path to the dataset")
    parser.add_argument("--epochs", type=int, default=500,
                       help="Number of epochs to train for")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of worker processes for data loading")
    parser.add_argument("--dataset_size", type=int, default=30000,
                       help="Number of images in the dataset to use for training")
    parser.add_argument("--use_pretrained", type=bool, default=False,
                       help="Flag to use pretrained models if available")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists("saved"):
        os.makedirs("saved")
    if not os.path.exists("saved/img"):
        os.makedirs("saved/img")

    if not os.path.exists(args.dataset_path):
        print("Please download and extract the dataset from the provided link.")
        exit()

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    if args.use_pretrained and os.path.exists("saved/generator.pt") and os.path.exists("saved/discriminator.pt"):
        generator.load_state_dict(torch.load("saved/generator.pt"))
        discriminator.load_state_dict(torch.load("saved/discriminator.pt"))
    else:
        generator.apply(weight_init)
        discriminator.apply(weight_init)

    dataset = AnimeDataset(os.getcwd(), args.dataset_path, args.dataset_size)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    loss_criterion = nn.BCELoss().to(device)
    optimizer_gen = Adam(generator.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
    optimizer_disc = Adam(discriminator.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))

    fixed_noise = torch.randn(64, 100, 1, 1, device=device)

    for epoch in range(args.epochs):
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
        for i, real_images in progress_bar:
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)

            # Train Discriminator
            discriminator.zero_grad()
            outputs_real = discriminator(real_images.to(device).float()).squeeze()
            loss_real = loss_criterion(outputs_real, real_labels)

            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_images = generator(noise)
            outputs_fake = discriminator(fake_images.detach()).squeeze()
            loss_fake = loss_criterion(outputs_fake, fake_labels)

            loss_discriminator = loss_real + loss_fake
            loss_discriminator.backward()
            optimizer_disc.step()

            # Train Generator
            generator.zero_grad()
            outputs_fake = discriminator(fake_images).squeeze()
            loss_generator = loss_criterion(outputs_fake, real_labels)
            loss_generator.backward()
            optimizer_gen.step()

            progress_bar.set_description(f"Epoch [{epoch+1}/{args.epochs}] Loss D: {loss_discriminator.item():.4f}, loss G: {loss_generator.item():.4f}")

        if epoch % 20 == 0 or epoch == args.epochs-1:

            # Saving a grid of generated images
            with torch.no_grad():
                fake_images = generator(fixed_noise)
                img_grid = make_grid(fake_images, normalize=True)
                save_path = f"saved/img/epoch_{epoch}.png"
                save_image(img_grid, save_path)
                # Saving model checkpoints
    torch.save(generator.state_dict(), f"saved/generator_epoch.pt")
    torch.save(discriminator.state_dict(), f"saved/discriminator_epoch.pt")
    create_gif("training_evolution.gif", "saved/img/")
    visualize_loss("generator_loss.txt", "discriminator_loss.txt")

if __name__ == "__main__":
    main()

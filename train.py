import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl import Discriminator
from torchvision.utils import save_image
from torchvision import transforms

# Define the training function
def train(train_loader, generator_pipeline, discriminator, optimizer_G, optimizer_D, criterion_GAN, criterion_structure, criterion_content, device, num_epochs=10):
    generator_pipeline.train()  # Set generator to training mode
    discriminator.train()  # Set discriminator to training mode

    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(tqdm(train_loader)):
            real_images = real_images.to(device)

            # ---------------------
            # 1. Generator Training
            # ---------------------
            optimizer_G.zero_grad()

            # Generate images using the generator pipeline
            generated_images = generator_pipeline(real_images)

            # Adversarial loss (for generator)
            real_labels = torch.ones(generated_images.size(0), 1, device=device)
            pred_fake = discriminator(generated_images)
            loss_GAN = criterion_GAN(pred_fake, real_labels)

            # Content loss
            loss_content = criterion_content(generated_images, real_images)

            # Structure preservation loss
            loss_structure = criterion_structure(generated_images, real_images)

            # Total generator loss
            loss_G = loss_GAN + 0.8 * loss_content + 0.5 * loss_structure
            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            # 2. Discriminator Training
            # ---------------------
            optimizer_D.zero_grad()

            # Discriminator on real images
            real_labels = torch.ones(real_images.size(0), 1, device=device)
            pred_real = discriminator(real_images)
            loss_real = criterion_GAN(pred_real, real_labels)

            # Discriminator on generated (fake) images
            fake_labels = torch.zeros(generated_images.size(0), 1, device=device)
            pred_fake = discriminator(generated_images.detach())
            loss_fake = criterion_GAN(pred_fake, fake_labels)

            # Total discriminator loss
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            # Print loss every 100 iterations
            if i % 100 == 0:
                print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss D: {loss_D.item()}, Loss G: {loss_G.item()}')

        # Save generated images after each epoch
        save_image(generated_images * 0.5 + 0.5, os.path.join('output_images', f'epoch_{epoch}.png'))

# Define the main function
def main():
    # Set the device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the generator pipeline
    model_path = '/path/to/your/model'  # Path to the pretrained MasaCtrlPipeline model
    generator_pipeline = MasaCtrlPipeline.from_pretrained(model_path).to(device)

    # Initialize the discriminator
    discriminator = Discriminator().to(device)

    # Define optimizers
    optimizer_G = torch.optim.Adam(generator_pipeline.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Define loss functions
    criterion_GAN = torch.nn.BCELoss()
    criterion_structure = torch.nn.MSELoss()
    criterion_content = torch.nn.L1Loss()

    # Define dataset transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Create dataset and DataLoader
    train_dataset = ...  # Replace with your actual dataset
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Train the model
    train(train_loader, generator_pipeline, discriminator, optimizer_G, optimizer_D, criterion_GAN, criterion_structure, criterion_content, device)

if __name__ == "__main__":
    main()
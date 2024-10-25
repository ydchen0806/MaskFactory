import os
import torch
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.utils import save_image
from masactrl.diffuser_utils import MasaCtrlPipeline
from tqdm import tqdm

# Function to load and prepare the input image for inference
def load_and_prepare_image(image_path, device):
    """
    Load and prepare an image from a given path.
    """
    image = read_image(image_path).float() / 255.0  # Normalize to [0, 1]
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)  # Convert grayscale to RGB
    image = F.interpolate(image.unsqueeze(0), (512, 512))  # Resize the image
    return image.to(device)

# Function for inference to generate images
def infer(generator_pipeline, image_path, output_dir, device):
    """
    Perform inference and generate new images.
    """
    generator_pipeline.eval()  # Set the generator pipeline to evaluation mode
    os.makedirs(output_dir, exist_ok=True)

    # Load and prepare the image
    image = load_and_prepare_image(image_path, device)

    # Generate image using the generator pipeline
    with torch.no_grad():
        generated_image = generator_pipeline(image)

    # Save the generated image
    save_image(generated_image * 0.5 + 0.5, os.path.join(output_dir, 'generated_image.png'))
    print(f"Generated image saved at {output_dir}")

# Define the main function for inference
def main():
    # Set the device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pretrained generator pipeline
    model_path = '/path/to/your/model'  # Path to the pretrained MasaCtrlPipeline model
    generator_pipeline = MasaCtrlPipeline.from_pretrained(model_path).to(device)

    # Perform inference
    image_path = "path_to_input_image.png"  # Replace with the actual input image path
    output_dir = "output_images"
    infer(generator_pipeline, image_path, output_dir, device)

if __name__ == "__main__":
    main()
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms


def visualize_reconstruction(model,  image_path, transform, device='cuda'):
    """
    Visualizes the original and reconstructed image from the model.

    Args:
    - model (nn.Module): Trained autoencoder model.
    - image_tensor (torch.Tensor): 4D tensor of shape (1, C, H, W).
    - device (str): Device to which model and data should be moved before inference. Default: 'cuda'.

    Returns:
    - reconstruction_error (float): Mean Squared Error between the original and reconstructed image.
    """
    model.eval()  # Set model to evaluation mode

    # Load the image and apply transformations
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    # Get the reconstructed image
    with torch.no_grad():
        reconstructed_tensor = model(image_tensor)

    # Compute the reconstruction error (MSE)
    mse_loss = torch.nn.functional.mse_loss(image_tensor, reconstructed_tensor)
    reconstruction_error = mse_loss.item()

    # Convert tensors to numpy arrays for visualization
    original_image = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    reconstructed_image = reconstructed_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

    # Assuming images were normalized to [-1, 1], denormalize for visualization
    original_image = (original_image * 0.5) + 0.5
    reconstructed_image = (reconstructed_image * 0.5) + 0.5

    # Visualization using Matplotlib
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image)
    plt.title('Reconstructed Image')
    plt.axis('off')

    plt.show()

    return reconstruction_error

def get_transform():
    """
    Gives the transform that needs to be applied on the image before prediction
    :return:
    """
    # transform = transforms.Compose([
    #     transforms.Resize((64, 64)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    return transform

def get_transform_v2():
    """
    Gives the transform that needs to be applied on the image before prediction
    :return:
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to 264x264
        transforms.ToTensor(),  # Convert PIL image to PyTorch tensor

    ])
    return transform

def get_transform_pretrained():
    """
    Gives the transform that needs to be applied on the image before prediction
    :return:
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
        transforms.ToTensor(),  # Convert the PIL image to a PyTorch Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])
    return transform
def get_construction_error_from_model(model, image, transform):

    model.eval()  # Set model to evaluation mode

    # Load the image and apply transformations
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)
    # Get the reconstructed image
    with torch.inference_mode():
        reconstructed_tensor = model(image_tensor)

    # Compute the reconstruction error (MSE)
    mse_loss = torch.nn.functional.mse_loss(image_tensor, reconstructed_tensor)
    reconstruction_error = mse_loss.item()

    # # Convert tensors to numpy arrays for visualization
    # original_image = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    # reconstructed_image = reconstructed_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    return reconstruction_error

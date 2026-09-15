import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Computes Grad-CAM for a given PyTorch model.
    Assumes the model has `extract_features` and `forward_from_features` methods.
    """
    def __init__(self, model):
        self.model = model
        self.feature_maps = None
        self.gradients = None
        
        self.model.eval()

    def _save_gradient(self, grad):
        self.gradients = grad

    def generate(self, input_tensor: torch.Tensor, target_class: int = None):
        """Generates Grad-CAM heatmap for a specific target class using input gradients."""
        # Ensure model gradients are cleared
        self.model.zero_grad()
        # Ensure input tensor tracks gradients
        if not input_tensor.requires_grad:
            input_tensor.requires_grad = True
        # Forward pass to get logits
        logits = self.model(input_tensor)
        # Determine target class if not provided
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        # Backward pass for target class
        logits[0, target_class].backward(retain_graph=True)
        # Gradient of the input
        grads = input_tensor.grad  # shape (B, C, H, W)
        # Compute CAM by averaging absolute gradients over channels
        cam = torch.mean(torch.abs(grads), dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.squeeze().detach().cpu().numpy()
        return cam, target_class

def overlay_cam(image_rgb: np.ndarray, cam: np.ndarray, colormap=cv2.COLORMAP_JET, alpha=0.5):
    """
    Overlays the Grad-CAM heatmap on the original image.
    """
    # Resize CAM to match image dimensions
    h, w = image_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # Convert CAM to heatmap
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    
    # Overlay
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, heatmap, alpha, 0)
    return overlay

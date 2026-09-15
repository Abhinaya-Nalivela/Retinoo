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
        """
        Generates Grad-CAM heatmap for a specific target class.
        """
        # Ensure model tracks gradients for Grad-CAM
        self.model.zero_grad()
        
        # 1. Forward pass to get feature maps
        features = self.model.extract_features(input_tensor)
        features.retain_grad()
        features.register_hook(self._save_gradient)
        self.feature_maps = features
        
        # 2. Forward pass from features to logits
        logits = self.model.forward_from_features(features)
        
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
            
        # 3. Backward pass for target class
        score = logits[0, target_class]
        score.backward(retain_graph=True)
        
        # 4. Compute CAM
        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted sum of feature maps
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        cam = F.relu(cam)  # Apply ReLU
        
        # Normalize between 0 and 1
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        
        cam = cam.squeeze()
        cam = cam.detach().cpu().tolist()
        cam = np.array(cam)
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

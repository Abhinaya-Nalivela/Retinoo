import torch
import os
from pathlib import Path

# Fix relative imports by modifying sys.path to run standalone
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from python.classification.model import DRClassifier
from python.structures.vessels import UNet

def generate_dummy_models():
    print("Generating dummy models for end-to-end execution testing...")
    
    # 1. Classification Model (ONNX export for MATLAB)
    print("Generating DR Classifier...")
    cls_model = DRClassifier(num_classes=5, pretrained=False)
    # Don't train, just export random weights
    cls_model.eval()
    
    cls_save_dir = Path("models/classification")
    cls_save_dir.mkdir(parents=True, exist_ok=True)
    pth_path = cls_save_dir / "dr_classifier.pth"
    torch.save(cls_model.state_dict(), pth_path)
    print(f"Saved PyTorch classifier weights to {pth_path}")

    # Export ONNX for MATLAB Deep Learning Toolbox
    onnx_path = cls_save_dir / "dr_classifier.onnx"
    try:
        dummy_input = torch.randn(1, 3, 512, 512)
        torch.onnx.export(
            cls_model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_fundus"],
            output_names=["logits"],
            dynamo=False,
        )
        print(f"Exported ONNX classifier to {onnx_path}")
    except Exception as e:
        print(f"Note: ONNX export skipped ({e}). PyTorch .pth weights are available.")
    
    # 2. U-Net models for Lesions and Vessels (.pth files)
    print("Generating Segmentation Models...")
    seg_model = UNet(n_channels=3, n_classes=1)
    
    struct_dir = Path("models/structures")
    struct_dir.mkdir(parents=True, exist_ok=True)
    torch.save(seg_model.state_dict(), struct_dir / "best_unet_vessels.pth")
    print(f"Saved dummy vessel model.")
    
    lesion_dir = Path("models/lesions")
    lesion_dir.mkdir(parents=True, exist_ok=True)
    for lt in ["MA", "HE", "EX"]:
        torch.save(seg_model.state_dict(), lesion_dir / f"best_unet_{lt}.pth")
        print(f"Saved dummy {lt} model.")

if __name__ == "__main__":
    generate_dummy_models()

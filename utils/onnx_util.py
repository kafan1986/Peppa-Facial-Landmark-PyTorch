import torch
import torch.onnx
from models.slim import Slim

x = torch.randn(1, 3, 96, 96)
model = Slim()
model.load_state_dict(torch.load("./weights/slim96_epoch_37_0.1151.pth", map_location="cpu"))
model.eval()
torch.onnx.export(model, x, "./weights/slim_96_latest.onnx", input_names=["input1"], output_names=['output1'])

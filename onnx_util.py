import torch
import torch.onnx
from models.slim import Slim
from collections import OrderedDict
x = torch.randn(1, 3, 96, 96)
state_dict = torch.load("./weights/slim96_epoch_37_0.1151.pth")
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
model = Slim()
model.load_state_dict(new_state_dict)
model.eval()
torch.onnx.export(model, x, "./weights/slim_96_latest.onnx", input_names=["input1"], output_names=['output1'])

import  torch
import torchvision
from PIL import Image
from torchvision.transforms import ToTensor
import utils
import onnxruntime as runtime
from PIL import Image
import numpy as np

def preprocess_image(image_path):
    # 打开并调整图片大小
    img = Image.open(image_path).convert("RGB").resize((1280, 1280))
    img_data = np.array(img, dtype=np.float32) / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))  # 将HWC转为CHW
    img_data = np.expand_dims(img_data, axis=0)

    return img_data

model_path="./out/s2000t.onnx"
image_path="./out/img_T.png"

input_data=preprocess_image(image_path)

model = runtime.InferenceSession(model_path)

input_name = model.get_inputs()[0].name
outputs = model.run(None, {input_name: input_data})

print(outputs.shape)





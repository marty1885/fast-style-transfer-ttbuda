import pybuda
import onnx
import torch
import PIL
import torchvision

if __name__ == '__main__':
    model_path = "pointilism-9.onnx"

    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"


    onnx = onnx.load(model_path)
    buda = pybuda.OnnxModule("FastStyleTransfer", onnx, model_path)

    tt0 = pybuda.TTDevice("tt0")
    tt0.place_module(buda)

    image_path = "test.jpg"
    image = PIL.Image.open(image_path).resize((224, 224))
    # Convert to NCHW RGB
    image = torchvision.transforms.ToTensor()(image).unsqueeze(0)

    print("===================== Running inference =====================")
    tt0.push_to_inputs([image])
    print("==================== Stage 1 ====================")
    outqueue = pybuda.run_inference()
    print("==================== Stage 2 ====================")
    out = outqueue.get()
    print("==================== Stage 3 ====================")

    print(out)

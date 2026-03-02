# CPPN Trainer (AMD Supported)

This repository contains a simple trainer for Compositional Pattern-Producing Networks (CPPNs). It is built with PyTorch and is compatible with AMD GPUs out-of-the-box via DirectML (though it will also run normally on NVIDIA/CUDA and CPU setups).

The goal of this project is to take a target image, train a neural network to reproduce that image pixel-by-pixel as a continuous mathematical function, and then export those trained neural network weights natively into raw GLSL code. 

## How it works

1. **Train**: You run the Python training script and select a target image. The script uses PyTorch to train a multi-layer Neural Network (the CPPN) to memorize the colors of your image based purely on screen coordinates `(x, y)`.
2. **Export**: Once you press 'Q' to quit training, the script extracts the network's weights and biases and bakes them directly into a Fragment Shader format.
3. **View**: It outputs a file named `trained_cppn.glsl`. You can open this file in the provided `GLSL_viewer.py` to see the neural network being evaluated entirely on your GPU inside an OpenGL fragment shader in real-time.

## Usage

### 1. Training the CPPN

Run the training script:

```bash
python train_cppn.py
```

- A file dialog will open asking you for an image.
- Set your preferred image, and training will immediately begin. You will see a live updating split-screen OpenCV window comparing your target image with the network's current guess. 
- You can dynamically adjust the `Learning Rate` via the open OpenCV trackbar. (If loss stops going down, lower the learning rate!)
- When you are satisfied with how the network looks, press **`Q`** while your mouse is over the OpenCV window.
- The script will exit and generate `trained_cppn.glsl` in the current folder.

### 2. Viewing the Trained GLSL

After training, you can view your new shader visually utilizing the included OpenGL script:

```bash
python GLSL_viewer.py
```

- A file dialog will appear. Select the newly minted `trained_cppn.glsl` (or any other `.frag`/`.glsl` files you want).
- It will open a PyGame window rendering your fragment shader dynamically!

### Requirements

See `requirements.txt` for standard dependencies. If you are on an AMD GPU on Windows, the script will automatically attempt to use `torch-directml` for GPU-accelerated training.

## License

This project is open-source and licensed under the [MIT License](LICENSE).

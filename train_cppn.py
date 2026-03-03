import argparse
import sys
import os
import math
import tkinter as tk
from tkinter import filedialog

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from PIL import Image
    import numpy as np
    import cv2
except ImportError:
    print("Missing dependencies. Please install them by running:")
    print("pip install torch torchvision numpy opencv-python pillow")
    sys.exit(1)

try:
    import torch_directml
    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False

# =============================================================================
# 1. PyTorch replica of the GLSL Architecture
# =============================================================================
# This class defines a Compositional Pattern Producing Network (CPPN).
# The architecture is designed to map exactly to the provided GLSL shader.
# It uses 8 input features, separated logically into specific tensor sizes 
# to manually replicate how typical graphics shaders process vec4 arrays.
class GLSL_CPPN(nn.Module):
    def __init__(self):
        super().__init__()
        # --- Hidden Layer 1 & 2 ---
        # The first sets of layers read from 'in_buf[6]' and 'in_buf[7]' in the GLSL code,
        # which represent our 8 spatial and mathematical input features.
        self.l_buf0 = nn.Linear(8, 4)
        self.l_buf1 = nn.Linear(8, 4)
        self.l_buf2 = nn.Linear(8, 4)
        self.l_buf3 = nn.Linear(8, 4)
        
        # --- Deep Layer 1 ---
        # These layers concatenate the outputs of the first hidden layers (buf0 to buf3),
        # acting on 16 combined features.
        self.l_buf4 = nn.Linear(16, 4)
        self.l_buf5 = nn.Linear(16, 4)
        
        # --- Deep Layer 2 ---
        # These layers take the 16 features from the previous layer plus the 8 new 
        # outputs from buf4 and buf5, acting on 24 combined features.
        self.l_buf6 = nn.Linear(24, 4)
        self.l_buf7 = nn.Linear(24, 4)
        
        # --- Output Layer ---
        # The final layer aggregates all 32 features generated across the network
        # and outputs 4 values (typically RGBA, though only RGB is used for the loss).
        self.l_out = nn.Linear(32, 4)

    def forward(self, input_features):
        # input_features: Tensor of shape (..., 8) representing the coordinate grid
        buf6_7 = input_features
        
        # Pass inputs through the first set of hidden layers with Sigmoid activation
        buf0 = torch.sigmoid(self.l_buf0(buf6_7))
        buf1 = torch.sigmoid(self.l_buf1(buf6_7))
        buf2 = torch.sigmoid(self.l_buf2(buf6_7))
        buf3 = torch.sigmoid(self.l_buf3(buf6_7))
        
        # Concatenate outputs mapping to in_buf[0..3]
        buf0_3 = torch.cat([buf0, buf1, buf2, buf3], dim=-1)
        
        # Pass through Deep Layer 1
        buf4 = torch.sigmoid(self.l_buf4(buf0_3))
        buf5 = torch.sigmoid(self.l_buf5(buf0_3))
        
        # Concatenate the new outputs with the old ones (mapping to in_buf[0..5])
        buf0_5 = torch.cat([buf0_3, buf4, buf5], dim=-1)
        
        # Pass through Deep Layer 2
        buf6 = torch.sigmoid(self.l_buf6(buf0_5))
        buf7 = torch.sigmoid(self.l_buf7(buf0_5))
        
        # Combine all features for the final output layer
        buf0_7 = torch.cat([buf0_5, buf6, buf7], dim=-1)
        
        # Final color computation
        out = torch.sigmoid(self.l_out(buf0_7))
        return out

# =============================================================================
# 2. Input Preprocessing: From Grid to Target Image
# =============================================================================
# Reads a desired target image, creates an explicit X, Y coordinate grid scaled
# from -1 to 1, and assigns mathematical characteristics like radial distance, 
# zero vectors, and noise initialization states that mimic shader behavior.
def get_grid_and_target(image_path, size=(128, 128)):
    # 1. Load the target image and resize mapping RGB directly
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size, Image.Resampling.LANCZOS)
    target = np.array(img, dtype=np.float32) / 255.0

    # 2. Generate Coordinate Features Mapping
    # Creating an evenly spaced coordinate space. 
    # y = np.linspace(-1, 1, size[1]) matches flipped native space in shaders.
    x = np.linspace(-1, 1, size[0])
    y = np.linspace(-1, 1, size[1])  
    xx, yy = np.meshgrid(x, y)
    
    # 3. Simulate GLSL input arguments for the first set of inputs (in_buf[6])
    # x, y coordinates alongside two static properties replicating random 
    # noise fields generated internally in the shader code
    buf6_0 = xx
    buf6_1 = yy
    buf6_2 = np.full_like(xx, 0.3948) # Default internal noise seed 0
    buf6_3 = np.full_like(xx, 0.36)   # Default internal noise seed 1
    
    # Simulate GLSL input arguments for the second set of inputs (in_buf[7])
    # The first variable holds an extra noise variable, the second computes the 
    # polar distance from the center (sqrt(x^2 + y^2)), followed by padded zeros.
    buf7_0 = np.full_like(xx, 0.14)   # Default internal noise seed 2
    buf7_1 = np.sqrt(xx**2 + yy**2)   # Radial distance feature
    buf7_2 = np.zeros_like(xx)
    buf7_3 = np.zeros_like(xx)
    
    # Pack these 8 features together perfectly conforming to GLSL expectations
    inputs = np.stack([buf6_0, buf6_1, buf6_2, buf6_3, buf7_0, buf7_1, buf7_2, buf7_3], axis=-1)
    
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

# =============================================================================
# 3. Model Export: Translating PyTorch Weights to GLSL
# =============================================================================
# Reads out all learned parameters from the PyTorch linear layers,
# converts them into explicit vec4 and mat4 representations, and
# injects them directly into a pre-written shader template file.
def export_weights(model, filename="trained_cppn.glsl"):
    # Convert consecutive values into formatting string representing vec4 mapping
    def format_vec4(vec):
        return f"vec4({vec[0]:.6f},{vec[1]:.6f},{vec[2]:.6f},{vec[3]:.6f})"

    # Iterative method to compose a whole mat4 (4 vec4 column vectors)
    def format_mat4(mat):
        c0, c1, c2, c3 = [format_vec4(mat[:, i]) for i in range(4)]
        return f"mat4({c0},{c1},{c2},{c3})"

    # Detach gradients natively mapping PyTorch Linear layers weight matrices and bias arrays
    def get_layer(layer):
        return layer.weight.detach().cpu().numpy(), layer.bias.detach().cpu().numpy()

    # Get explicitly referenced weight vectors mapped exactly back 
    # individually matching their corresponding layers mapped inside the shader array.
    W1, b1 = get_layer(model.l_buf0)
    W2, b2 = get_layer(model.l_buf1)
    W3, b3 = get_layer(model.l_buf2)
    W4, b4 = get_layer(model.l_buf3)
    
    # Each input expands to process different combinations of buffers
    W5, b5 = get_layer(model.l_buf4)
    W6, b6 = get_layer(model.l_buf5)
    W7, b7 = get_layer(model.l_buf6)
    W8, b8 = get_layer(model.l_buf7)
    W9, b9 = get_layer(model.l_out)

    glsl_template = """

varying vec2 v_TexCoord;
uniform float g_Time;
uniform vec3 g_Screen;
uniform float uHueShift; // {{"material":"uHueShift","default":1,"range":[0,600]}}
uniform float uNoise; // {{"material":"uNoise","default":1,"range":[0,1]}}
uniform float uScan; // {{"material":"uScan","default":1,"range":[0,1]}}
uniform float uScanFreq; // {{"material":"uScanFreq","default":1,"range":[0,1]}}
uniform float uWarp; // {{"material":"uWarp","default":1,"range":[0,1]}}
#define uTime g_Time
#define uResolution g_Screen

vec4 mmul(mat4 m, vec4 v) {{
    return v.x*m[0] + v.y*m[1] + v.z*m[2] + v.w*m[3];
}}
vec3 mmul(mat3 m, vec3 v) {{
    return v.x*m[0] + v.y*m[1] + v.z*m[2];
}}

float hash(vec2 p) {{
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}}

// 2D Value Noise using Hermite interpolation
float hermiteNoise(vec2 uv) {{
    vec2 i = floor(uv);  // Integer grid coordinates
    vec2 f = fract(uv);  // Fractional coordinates

    // Cubic Hermite interpolation polynomial: 3f^2 - 2f^3
    vec2 u = f * f * (3.0 - 2.0 * f);

    // Get random values for the 4 corners of the grid cell
    float a = hash(i + vec2(0.0, 0.0));
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    // Bilinearly interpolate between the corners using the Hermite curve
    return mix(mix(a, b, u.x), 
               mix(c, d, u.x), u.y);
}}

float rand(vec2 c){{return fract(sin(dot(c,vec2(12.9898,78.233)))*43758.5453);}}

const mat3 rgb2yiq=mat3(0.299,0.596,0.211,0.587,-0.274,-0.523,0.114,-0.322,0.312);
const mat3 yiq2rgb=mat3(1.0,0.956,0.621,1.0,-0.272,-0.647,1.0,-1.106,1.703);

vec3 hueShiftRGB(vec3 col,float deg){{
    vec3 yiq=mmul(rgb2yiq, col);
    float rad=radians(deg);
    float ch=cos(rad),sh=sin(rad);
    vec3 yiqShift=vec3(yiq.x,yiq.y*ch-yiq.z*sh,yiq.y*sh+yiq.z*ch);
    return clamp(mmul(yiq2rgb, yiqShift),0.0,1.0);
}}

vec4 sigmoid(vec4 x){{return 1./(1.+exp(-x));}}

vec4 cppn_fn(vec2 coordinate, float in0, float in1, float in2) {{
    // --- INPUT LAYER ---
    vec4 in_buf[8];
    in_buf[6] = vec4(coordinate.x, coordinate.y, 0.3948 + in0, 0.36 + in1);
    in_buf[7] = vec4(0.14 + in2, sqrt(coordinate.x * coordinate.x + coordinate.y * coordinate.y), 0., 0.);
    
    // --- WEIGHTS & BIASES ---
    // Hidden Layer 1
    mat4 w1_1 = {w1_1}; mat4 w1_2 = {w1_2}; vec4 b1 = {b1};
    mat4 w2_1 = {w2_1}; mat4 w2_2 = {w2_2}; vec4 b2 = {b2};

    // Hidden Layer 2
    mat4 w3_1 = {w3_1}; mat4 w3_2 = {w3_2}; vec4 b3 = {b3};
    mat4 w4_1 = {w4_1}; mat4 w4_2 = {w4_2}; vec4 b4 = {b4};

    // Deep Layer 1
    mat4 w5_1 = {w5_1}; mat4 w5_2 = {w5_2}; mat4 w5_3 = {w5_3}; mat4 w5_4 = {w5_4}; vec4 b5 = {b5};
    mat4 w6_1 = {w6_1}; mat4 w6_2 = {w6_2}; mat4 w6_3 = {w6_3}; mat4 w6_4 = {w6_4}; vec4 b6 = {b6};

    // Deep Layer 2
    mat4 w7_1 = {w7_1}; mat4 w7_2 = {w7_2}; mat4 w7_3 = {w7_3}; mat4 w7_4 = {w7_4}; mat4 w7_5 = {w7_5}; mat4 w7_6 = {w7_6}; vec4 b7 = {b7};
    mat4 w8_1 = {w8_1}; mat4 w8_2 = {w8_2}; mat4 w8_3 = {w8_3}; mat4 w8_4 = {w8_4}; mat4 w8_5 = {w8_5}; mat4 w8_6 = {w8_6}; vec4 b8 = {b8};

    // Output Layer
    mat4 w9_1 = {w9_1}; mat4 w9_2 = {w9_2}; mat4 w9_3 = {w9_3}; mat4 w9_4 = {w9_4}; mat4 w9_5 = {w9_5}; mat4 w9_6 = {w9_6}; mat4 w9_7 = {w9_7}; mat4 w9_8 = {w9_8}; vec4 b9 = {b9};

    // --- APPLYING NEURAL NETWORK ---
    // Hidden Layer 1 calculation
 
    in_buf[0] = mmul(w1_1, in_buf[6]) + mmul(w1_2, in_buf[7]) + b1;
    in_buf[1] = mmul(w2_1, in_buf[6]) + mmul(w2_2, in_buf[7]) + b2;
    in_buf[0] = sigmoid(in_buf[0]); in_buf[1] = sigmoid(in_buf[1]);
    
    // Hidden Layer 2 calculation
    in_buf[2] = mmul(w3_1, in_buf[6]) + mmul(w3_2, in_buf[7]) + b3;
    in_buf[3] = mmul(w4_1, in_buf[6]) + mmul(w4_2, in_buf[7]) + b4;
    in_buf[2] = sigmoid(in_buf[2]); in_buf[3] = sigmoid(in_buf[3]);

    // Deep Layer 1 calculation
    in_buf[4] = mmul(w5_1, in_buf[0]) + mmul(w5_2, in_buf[1]) + mmul(w5_3, in_buf[2]) + mmul(w5_4, in_buf[3]) + b5;
    in_buf[5] = mmul(w6_1, in_buf[0]) + mmul(w6_2, in_buf[1]) + mmul(w6_3, in_buf[2]) + mmul(w6_4, in_buf[3]) + b6;
    in_buf[4] = sigmoid(in_buf[4]); in_buf[5] = sigmoid(in_buf[5]);
    
    // Deep Layer 2 calculation
    in_buf[6] = mmul(w7_1, in_buf[0]) + mmul(w7_2, in_buf[1]) + mmul(w7_3, in_buf[2]) + mmul(w7_4, in_buf[3]) + mmul(w7_5, in_buf[4]) + mmul(w7_6, in_buf[5]) + b7;
    in_buf[7] = mmul(w8_1, in_buf[0]) + mmul(w8_2, in_buf[1]) + mmul(w8_3, in_buf[2]) + mmul(w8_4, in_buf[3]) + mmul(w8_5, in_buf[4]) + mmul(w8_6, in_buf[5]) + b8;
    in_buf[6] = sigmoid(in_buf[6]); in_buf[7] = sigmoid(in_buf[7]);
    
    // Output Layer calculation
    in_buf[0] = mmul(w9_1, in_buf[0]) + mmul(w9_2, in_buf[1]) + mmul(w9_3, in_buf[2]) + mmul(w9_4, in_buf[3]) + mmul(w9_5, in_buf[4]) + mmul(w9_6, in_buf[5]) + mmul(w9_7, in_buf[6]) + mmul(w9_8, in_buf[7]) + b9;
    in_buf[0] = sigmoid(in_buf[0]);
    
    return vec4(in_buf[0].x, in_buf[0].y, in_buf[0].z, 1.);
}}

void preProcess(out vec4 fragColor,in vec2 fragCoord){{
    vec2 uv=fragCoord/uResolution.xy*2.-1.;
    uv.y*=-1.;
    uv+=uWarp*vec2(sin(uv.y*6.283+uTime*0.5),cos(uv.x*6.283+uTime*0.5))*0.05;
    fragColor=cppn_fn(uv,0.1*hermiteNoise(uv+0.3*uTime),0.1*hermiteNoise(uv+0.69*uTime),0.1*hermiteNoise(uv+0.44*uTime));
}}

void main(){{
    vec2 fragCoord = v_TexCoord.xy * uResolution.xy;
    
    vec4 col;
    preProcess(col, fragCoord);
    col.rgb=hueShiftRGB(col.rgb,uHueShift);
    float scanline_val=sin(fragCoord.y*uScanFreq)*0.5+0.5;
    col.rgb*=1.-(scanline_val*scanline_val)*uScan;
    col.rgb+=(rand(fragCoord+uTime)-0.5)*uNoise;
    gl_FragColor=vec4(clamp(col.rgb,0.0,1.0),1.0);
}}"""

    code = glsl_template.format(
        w1_1=format_mat4(W1[:, 0:4]), w1_2=format_mat4(W1[:, 4:8]), b1=format_vec4(b1),
        w2_1=format_mat4(W2[:, 0:4]), w2_2=format_mat4(W2[:, 4:8]), b2=format_vec4(b2),
        w3_1=format_mat4(W3[:, 0:4]), w3_2=format_mat4(W3[:, 4:8]), b3=format_vec4(b3),
        w4_1=format_mat4(W4[:, 0:4]), w4_2=format_mat4(W4[:, 4:8]), b4=format_vec4(b4),
        w5_1=format_mat4(W5[:, 0:4]), w5_2=format_mat4(W5[:, 4:8]), w5_3=format_mat4(W5[:, 8:12]), w5_4=format_mat4(W5[:, 12:16]), b5=format_vec4(b5),
        w6_1=format_mat4(W6[:, 0:4]), w6_2=format_mat4(W6[:, 4:8]), w6_3=format_mat4(W6[:, 8:12]), w6_4=format_mat4(W6[:, 12:16]), b6=format_vec4(b6),
        w7_1=format_mat4(W7[:, 0:4]), w7_2=format_mat4(W7[:, 4:8]), w7_3=format_mat4(W7[:, 8:12]), w7_4=format_mat4(W7[:, 12:16]), w7_5=format_mat4(W7[:, 16:20]), w7_6=format_mat4(W7[:, 20:24]), b7=format_vec4(b7),
        w8_1=format_mat4(W8[:, 0:4]), w8_2=format_mat4(W8[:, 4:8]), w8_3=format_mat4(W8[:, 8:12]), w8_4=format_mat4(W8[:, 12:16]), w8_5=format_mat4(W8[:, 16:20]), w8_6=format_mat4(W8[:, 20:24]), b8=format_vec4(b8),
        w9_1=format_mat4(W9[:, 0:4]), w9_2=format_mat4(W9[:, 4:8]), w9_3=format_mat4(W9[:, 8:12]), w9_4=format_mat4(W9[:, 12:16]), w9_5=format_mat4(W9[:, 16:20]), w9_6=format_mat4(W9[:, 20:24]), w9_7=format_mat4(W9[:, 24:28]), w9_8=format_mat4(W9[:, 28:32]), b9=format_vec4(b9)
    )
    
    with open(filename, "w") as f:
        f.write(code)
    print(f"\nSaved trained GLSL to: {filename}")

# =============================================================================
# 4. Main Training Routine: UI Control and Optimizer Loop
# =============================================================================
# - Parses CLI arguments. Selects hardware backend optimally (AMD, CUDA, CPU).
# - Generates live visualization utilizing OpenCV for direct user-evaluation.
# - Leverages learning rate decay tracking performance over step cycles via MSE.
def main():
    parser = argparse.ArgumentParser(description="Train a CPPN to match a picture and export to GLSL")
    parser.add_argument("image", nargs="?", help="Path to target image (optional, will open file dialog if omitted)")
    parser.add_argument("--size", type=int, default=256, help="Training resolution (larger = more GPU work, default: 256)")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate (default: 0.002)")
    parser.add_argument("--display-every", type=int, default=1000, help="Update preview every N steps (higher = faster training, default: 100)")
    parser.add_argument("--input-noise", type=float, default=0, help="Std of Gaussian noise added to inputs each step to prevent stagnation (default: 0.01, set 0 to disable)")
    parser.add_argument("--perturb-scale", type=float, default=0.02, help="Std of weight perturbation applied when LR scheduler reduces LR (default: 0.02, set 0 to disable)")
    args = parser.parse_args()

    image_path = args.image
    
    # 1. Provide an automated UI fallback using Tkinter if no image path is passed
    if not image_path:
        root = tk.Tk()
        root.withdraw()
        initial_dir = os.path.abspath(".")
        print("Please select an image in the file dialog...")
        image_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Select an image to train on",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.gif"),
                ("All files", "*.*")
            ]
        )
        if not image_path:
            print("No image selected. Exiting.")
            sys.exit(0)

    # 2. Check explicitly for available backends minimizing CPU bottleneck
    # - NVIDIA CUDA is preferred natively 
    # - Then looks for AMD hardware by checking the `torch_directml` driver status.
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif HAS_DIRECTML and torch_directml.is_available():
        device = torch_directml.device()
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    print(f"Loading image {image_path} at {args.size}x{args.size}...")
    
    # Generate spatial features mapping and correct target format representing the image
    inputs, targets = get_grid_and_target(image_path, size=(args.size, args.size))
    
    # Flatten inputs targeting PyTorch linear model batch configurations mapping
    inputs_flat = inputs.view(-1, 8).to(device)
    targets_flat = targets.view(-1, 3).to(device)
    
    # Initialize actual logical PyTorch replication module mapping GLSL
    model = GLSL_CPPN().to(device)

    # torch.compile gives a free speedup on CUDA (PyTorch 2.0+)
    # It dynamically detects specifically mapped `privateuseone` backends matching DirectML
    is_dml = False
    try:
        if 'privateuseone' in str(device):
            is_dml = True
    except:
        pass

    if not is_dml:
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception:
            pass  # Not available on this PyTorch version

    # 3. Setup explicitly mapping Optimizer dependencies targeting given backend
    # AMD drivers prefer RMSprop handling specific layer updates appropriately.
    if HAS_DIRECTML and is_dml:
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9) # type: ignore
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr) # type: ignore

    min_lr = 0.00001
    max_lr = 0.1

    # Apply a dynamic learning rate decay tracking `MSELoss()` flattening effectively
    # patience is in scheduler-step calls, not raw steps; min_lr floors it so it never reaches 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=min_lr)
    
    criterion = nn.MSELoss()
    
    print("\nStarting Training! Focus the image window and press 'q' to stop & save.")
    
    global current_lr
    current_lr = args.lr
    display_every = args.display_every
    
    # Handles dynamic mapping updates applying OpenCV tracks dynamically targeting optimizer features
    def on_lr_change(val):
        global current_lr
        val = max(0, min(100, val)) # Clamp to prevent out-of-bounds
        val_f = float(val) / 100.0
        current_lr = min_lr * (max_lr / min_lr) ** val_f
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
            
    # Calculate starting mapping UI offset mapping Logarithmic scales effectively
    init_lr = max(min_lr, min(max_lr, args.lr))
    init_val = int(round(100.0 * math.log(init_lr / min_lr) / math.log(max_lr / min_lr)))
    
    # At >=512 the window would be huge (2048px wide); halve it to keep it manageable
    display_scale = 2 if args.size >= 512 else 4

    step = 0
    prev_lr = optimizer.param_groups[0]['lr']
    try:
        while True:
            # Iterative Training Loop Logic mapping updates every single individual loop
            optimizer.zero_grad()
            
            # Input jitter: applied every 1001 steps randomly shifting noise mapping explicitly targeting input coordinates.
            # Used to automatically pull training state away from mapping local minimas stalling rendering effectiveness.
            if args.input_noise > 0 and step % 1001 == 0:
                noisy_inputs = inputs_flat + torch.randn_like(inputs_flat) * args.input_noise
                pred_flat = model(noisy_inputs)
            else:
                pred_flat = model(inputs_flat)
                
            # Compare generated RGB layers output via CPPN against flat normalized pixel array
            loss = criterion(pred_flat[:, :3], targets_flat) 
            loss.backward()
            optimizer.step()
            
            # Reconstruct image periodically for UI preview
            if step % display_every == 0:
                # Show window first so the trackbar has a valid parent window instance
                if step == 0:
                    # Initial empty setup so that setTrackbarPos doesn't throw a NULL window error
                    cv2.namedWindow("CPPN Trainer")
                    cv2.createTrackbar("Log10(LR)", "CPPN Trainer", init_val, 100, on_lr_change)
                
                # Update the scheduler every display_every steps with the current loss
                # (but only after the first 5000 steps allowing model to form basic structure mapping first)
                if step >= 5000:
                    scheduler.step(loss)
                
                # Fetch the current learning rate from the optimizer determining active loss modifications
                current_lr = optimizer.param_groups[0]['lr']

                # Weight perturbation: if scheduler just reduced LR, kick weights to escape plateau
                if args.perturb_scale > 0 and current_lr < prev_lr:
                    with torch.no_grad():
                        for param in model.parameters():
                            param.add_(torch.randn_like(param) * args.perturb_scale)
                    print(f"  [step {step}] LR reduced to {current_lr:.6f} — weight perturbation applied")
                prev_lr = current_lr
                
                # Stop updating UI if window was closed explicitly mapping system shutdown commands
                if cv2.getWindowProperty("CPPN Trainer", cv2.WND_PROP_VISIBLE) < 1:
                    print("\n\nTraining stopped by user (Window closed).")
                    break
                
                # Update the trackbar position to reflect the new learning rate correctly applying UI limitations mapping Log
                clamped_lr = max(min_lr, min(max_lr, current_lr))
                new_val = int(round(100.0 * math.log(clamped_lr / min_lr) / math.log(max_lr / min_lr)))
                new_val = max(0, min(100, new_val)) # Clamp to prevent GUI errors
                try:
                    cv2.setTrackbarPos("Log10(LR)", "CPPN Trainer", new_val)
                except cv2.error:
                    break
                
                # Render predicted mapping explicitly displaying real image and generated matrix output
                with torch.no_grad():
                    # Format picture for visual preview mapping pixel color array matrices correctly applying to cv2
                    pred_img = pred_flat.view(args.size, args.size, 4).cpu().numpy()
                    pred_rgb = (pred_img[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
                    target_rgb = (targets.numpy() * 255).clip(0, 255).astype(np.uint8)
                    
                    # Convert to BGR natively mapping OpenCV display matrices
                    pred_bgr = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)
                    target_bgr = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR)
                    
                    # Scale them up mapping target resolutions matching monitor outputs natively handling viewing scale sizes
                    display_bgr = np.hstack((target_bgr, pred_bgr))
                    display_bgr = cv2.resize(display_bgr, (args.size*display_scale, args.size*(display_scale//2)), interpolation=cv2.INTER_NEAREST)
                    
                    # Create a white text panel at the bottom displaying basic rendering tracking mapped output natively
                    panel_height = 80
                    text_panel = np.ones((panel_height, display_bgr.shape[1], 3), dtype=np.uint8) * 255
                    
                    # Stack the images and the text panel vertically
                    display_bgr = np.vstack((display_bgr, text_panel))
                    
                    # Add text to the new panel at the bottom formatting real-time updates natively mapping CV2 displays
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    status_text = f"Step: {step:05d} | Loss: {loss.item():.5f} | LR: {current_lr:.5f}"
                    
                    # Base Y position matches the scaled image height natively maintaining tracking scale formats mapping displays
                    base_y = args.size * (display_scale // 2)
                    
                    cv2.putText(display_bgr, status_text, (10, base_y + 30), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(display_bgr, "Press 'Q' to save & exit", (10, base_y + 60), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    
                    cv2.imshow("CPPN Trainer", display_bgr)
                    
                    # Stop if 'q' is pressed or Window is closed natively saving outputs executing correct GLSL export processes 
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        print("\n\nTraining stopped by user.")
                        break
            step += 1
            
    except KeyboardInterrupt:
        print("\n\nTraining stopped with Ctrl+C")
        
    cv2.destroyAllWindows()
    
    # Save the output weights generated during tracking dynamically mapped training loop natively
    export_weights(model, "trained_cppn.glsl")

if __name__ == "__main__":
    main()

import argparse
import sys
import os
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

# --- 1. exact PyTorch replica of the GLSL architecture ---
class GLSL_CPPN(nn.Module):
    def __init__(self):
        super().__init__()
        # Hidden 1 & 2 read buf[6] and buf[7] (inputs, size 8 total)
        self.l_buf0 = nn.Linear(8, 4)
        self.l_buf1 = nn.Linear(8, 4)
        self.l_buf2 = nn.Linear(8, 4)
        self.l_buf3 = nn.Linear(8, 4)
        
        # Deep 1 reads buf[0..3] (size 16 total)
        self.l_buf4 = nn.Linear(16, 4)
        self.l_buf5 = nn.Linear(16, 4)
        
        # Deep 2 reads buf[0..5] (size 24 total)
        self.l_buf6 = nn.Linear(24, 4)
        self.l_buf7 = nn.Linear(24, 4)
        
        # Output reads buf[0..7] (size 32 total)
        self.l_out = nn.Linear(32, 4)

    def forward(self, input_features):
        buf6_7 = input_features
        
        buf0 = torch.sigmoid(self.l_buf0(buf6_7))
        buf1 = torch.sigmoid(self.l_buf1(buf6_7))
        buf2 = torch.sigmoid(self.l_buf2(buf6_7))
        buf3 = torch.sigmoid(self.l_buf3(buf6_7))
        
        buf0_3 = torch.cat([buf0, buf1, buf2, buf3], dim=-1)
        
        buf4 = torch.sigmoid(self.l_buf4(buf0_3))
        buf5 = torch.sigmoid(self.l_buf5(buf0_3))
        
        buf0_5 = torch.cat([buf0_3, buf4, buf5], dim=-1)
        
        buf6 = torch.sigmoid(self.l_buf6(buf0_5))
        buf7 = torch.sigmoid(self.l_buf7(buf0_5))
        
        buf0_7 = torch.cat([buf0_5, buf6, buf7], dim=-1)
        
        out = torch.sigmoid(self.l_out(buf0_7))
        return out

# --- 2. Input Preprocessing ---
def get_grid_and_target(image_path, size=(128, 128)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size, Image.Resampling.LANCZOS)
    target = np.array(img, dtype=np.float32) / 255.0

    # Match the coordinate space of wall.frag
    x = np.linspace(-1, 1, size[0])
    y = np.linspace(-1, 1, size[1])  # Changed from 1, -1 to fix flipped coordinates natively for the app
    xx, yy = np.meshgrid(x, y)
    
    # EXACT inputs passed in wall.frag: 
    # buf[6] = vec4(coordinate.x, coordinate.y, 0.3948 + in0, 0.36 + in1);
    # buf[7] = vec4(0.14 + in2, sqrt(coordinate.x * coordinate.x + coordinate.y * coordinate.y), 0., 0.);
    buf6_0 = xx
    buf6_1 = yy
    buf6_2 = np.full_like(xx, 0.3948) # assuming in0=0
    buf6_3 = np.full_like(xx, 0.36)   # assuming in1=0
    
    buf7_0 = np.full_like(xx, 0.14)   # assuming in2=0
    buf7_1 = np.sqrt(xx**2 + yy**2)
    buf7_2 = np.zeros_like(xx)
    buf7_3 = np.zeros_like(xx)
    
    inputs = np.stack([buf6_0, buf6_1, buf6_2, buf6_3, buf7_0, buf7_1, buf7_2, buf7_3], axis=-1)
    
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

# --- 3. Exporter to GLSL ---
def export_weights(model, filename="trained_cppn.glsl"):
    def format_vec4(vec):
        return f"vec4({vec[0]:.6f},{vec[1]:.6f},{vec[2]:.6f},{vec[3]:.6f})"

    def format_mat4(mat):
        # We slice columns because GLSL constructs matrices column-major
        # e.g., mapping PyTorch Linear layers [out, in] accurately
        c0, c1, c2, c3 = [format_vec4(mat[:, i]) for i in range(4)]
        return f"mat4({c0},{c1},{c2},{c3})"

    def get_layer(layer):
        return layer.weight.detach().cpu().numpy(), layer.bias.detach().cpu().numpy()

    W1, b1 = get_layer(model.l_buf0)
    W2, b2 = get_layer(model.l_buf1)
    W3, b3 = get_layer(model.l_buf2)
    W4, b4 = get_layer(model.l_buf3)
    W5, b5 = get_layer(model.l_buf4)
    W6, b6 = get_layer(model.l_buf5)
    W7, b7 = get_layer(model.l_buf6)
    W8, b8 = get_layer(model.l_buf7)
    W9, b9 = get_layer(model.l_out)

    glsl_template = """#ifdef GL_ES
precision lowp float;
#endif

uniform float iTime;
uniform vec2 iResolution;
uniform float uHueShift;
uniform float uNoise;
uniform float uScan;
uniform float uScanFreq;
uniform float uWarp;
#define uTime iTime
#define uResolution iResolution

vec4 buf[8];
float rand(vec2 c){{return fract(sin(dot(c,vec2(12.9898,78.233)))*43758.5453);}}

mat3 rgb2yiq=mat3(0.299,0.587,0.114,0.596,-0.274,-0.322,0.211,-0.523,0.312);
mat3 yiq2rgb=mat3(1.0,0.956,0.621,1.0,-0.272,-0.647,1.0,-1.106,1.703);

vec3 hueShiftRGB(vec3 col,float deg){{
    vec3 yiq=rgb2yiq*col;
    float rad=radians(deg);
    float cosh=cos(rad),sinh=sin(rad);
    vec3 yiqShift=vec3(yiq.x,yiq.y*cosh-yiq.z*sinh,yiq.y*sinh+yiq.z*cosh);
    return clamp(yiq2rgb*yiqShift,0.0,1.0);
}}

vec4 sigmoid(vec4 x){{return 1./(1.+exp(-x));}}

vec4 cppn_fn(vec2 coordinate, float in0, float in1, float in2) {{
    // --- INPUT LAYER ---
    buf[6] = vec4(coordinate.x, coordinate.y, 0.3948 + in0, 0.36 + in1);
    buf[7] = vec4(0.14 + in2, sqrt(coordinate.x * coordinate.x + coordinate.y * coordinate.y), 0., 0.);
    
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
    buf[0] = w1_1 * buf[6] + w1_2 * buf[7] + b1;
    buf[1] = w2_1 * buf[6] + w2_2 * buf[7] + b2;
    buf[0] = sigmoid(buf[0]); buf[1] = sigmoid(buf[1]);
    
    // Hidden Layer 2 calculation
    buf[2] = w3_1 * buf[6] + w3_2 * buf[7] + b3;
    buf[3] = w4_1 * buf[6] + w4_2 * buf[7] + b4;
    buf[2] = sigmoid(buf[2]); buf[3] = sigmoid(buf[3]);

    // Deep Layer 1 calculation
    buf[4] = w5_1 * buf[0] + w5_2 * buf[1] + w5_3 * buf[2] + w5_4 * buf[3] + b5;
    buf[5] = w6_1 * buf[0] + w6_2 * buf[1] + w6_3 * buf[2] + w6_4 * buf[3] + b6;
    buf[4] = sigmoid(buf[4]); buf[5] = sigmoid(buf[5]);
    
    // Deep Layer 2 calculation
    buf[6] = w7_1 * buf[0] + w7_2 * buf[1] + w7_3 * buf[2] + w7_4 * buf[3] + w7_5 * buf[4] + w7_6 * buf[5] + b7;
    buf[7] = w8_1 * buf[0] + w8_2 * buf[1] + w8_3 * buf[2] + w8_4 * buf[3] + w8_5 * buf[4] + w8_6 * buf[5] + b8;
    buf[6] = sigmoid(buf[6]); buf[7] = sigmoid(buf[7]);
    
    // Output Layer calculation
    buf[0] = w9_1 * buf[0] + w9_2 * buf[1] + w9_3 * buf[2] + w9_4 * buf[3] + w9_5 * buf[4] + w9_6 * buf[5] + w9_7 * buf[6] + w9_8 * buf[7] + b9;
    buf[0] = sigmoid(buf[0]);
    
    return vec4(buf[0].x, buf[0].y, buf[0].z, 1.);
}}

void mainImage(out vec4 fragColor,in vec2 fragCoord){{
    vec2 uv=fragCoord/uResolution.xy*2.-1.;
    uv.y*=-1.;
    uv+=uWarp*vec2(sin(uv.y*6.283+uTime*0.5),cos(uv.x*6.283+uTime*0.5))*0.05;
    fragColor=cppn_fn(uv,0.1*sin(0.3*uTime),0.1*sin(0.69*uTime),0.1*sin(0.44*uTime));
}}

void main(){{
    vec4 col;mainImage(col,gl_FragCoord.xy);
    col.rgb=hueShiftRGB(col.rgb,uHueShift);
    float scanline_val=sin(gl_FragCoord.y*uScanFreq)*0.5+0.5;
    col.rgb*=1.-(scanline_val*scanline_val)*uScan;
    col.rgb+=(rand(gl_FragCoord.xy+uTime)-0.5)*uNoise;
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

# --- 4. Main Training Loop ---
def main():
    parser = argparse.ArgumentParser(description="Train a CPPN to match a picture and export to GLSL")
    parser.add_argument("image", nargs="?", help="Path to target image (optional, will open file dialog if omitted)")
    parser.add_argument("--size", type=int, default=128, help="Training resolution (smaller = faster, default: 128)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    args = parser.parse_args()

    image_path = args.image
    
    # If no image is provided via command line, open a file browser
    if not image_path:
        root = tk.Tk()
        root.withdraw() # Hide the main tk window
        print("Please select an image in the file dialog...")
        image_path = filedialog.askopenfilename(
            title="Select an image to train on",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.gif"),
                ("All files", "*.*")
            ]
        )
        # Check if user cancelled
        if not image_path:
            print("No image selected. Exiting.")
            sys.exit(0)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif HAS_DIRECTML and torch_directml.is_available():
        device = torch_directml.device()
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    print(f"Loading image {image_path} at {args.size}x{args.size}...")
    
    inputs, targets = get_grid_and_target(image_path, size=(args.size, args.size))
    
    # Flatten shapes for Dense Network batch training
    inputs_flat = inputs.view(-1, 8).to(device)
    targets_flat = targets.view(-1, 3).to(device)  # We train on RGB only, ignoring alpha
    
    model = GLSL_CPPN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    print("\nStarting Training! Focus the image window and press 'q' to stop & save.")
    
    # Store learning rate globally so slider callback can access it
    global current_lr
    current_lr = args.lr
    
    def on_lr_change(val):
        global current_lr
        # Map slider value [0, 100] to learning rate [0.0001, 0.1] logarithmically
        # val=0 -> 0.0001
        # val=50 -> 0.003
        # val=100 -> 0.1
        val_f = float(val) / 100.0
        min_lr = 0.0001
        max_lr = 0.1
        current_lr = min_lr * (max_lr / min_lr) ** val_f
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
            
    cv2.namedWindow("CPPN Trainer")
    # Initialize slider position based on args.lr
    import math
    min_lr = 0.0001
    max_lr = 0.1
    # clamp args.lr into bounds before log
    init_lr = max(min_lr, min(max_lr, args.lr))
    init_val = int(100.0 * math.log(init_lr / min_lr) / math.log(max_lr / min_lr))
    cv2.createTrackbar("Log10(LR)", "CPPN Trainer", init_val, 100, on_lr_change)
    
    step = 0
    try:
        while True:
            optimizer.zero_grad()
            pred_flat = model(inputs_flat)
            # Compare output with targets
            loss = criterion(pred_flat[:, :3], targets_flat) 
            loss.backward()
            optimizer.step()
            
            # Reconstruct image periodically for UI preview
            if step % 20 == 0:
                with torch.no_grad():
                    # Format picture for visual preview
                    pred_img = pred_flat.view(args.size, args.size, 4).cpu().numpy()
                    pred_rgb = (pred_img[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
                    target_rgb = (targets.numpy() * 255).clip(0, 255).astype(np.uint8)
                    
                    # Convert to BGR for OpenCV display
                    pred_bgr = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)
                    target_bgr = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR)
                    
                    # Scale them up so the window is easy to see
                    display_bgr = np.hstack((target_bgr, pred_bgr))
                    display_bgr = cv2.resize(display_bgr, (args.size*4, args.size*2), interpolation=cv2.INTER_NEAREST)
                    
                    # Add text to the image
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    status_text = f"Step: {step:05d} | Loss: {loss.item():.4f} | LR: {current_lr:.5f}"
                    cv2.putText(display_bgr, status_text, (10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(display_bgr, "Press 'Q' to save & exit", (10, 60), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    cv2.imshow("CPPN Trainer", display_bgr)
                    
                    # Stop if 'q' is pressed or Window is closed
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        print("\n\nTraining stopped by user.")
                        break
            step += 1
            
    except KeyboardInterrupt:
        print("\n\nTraining stopped with Ctrl+C")
        
    cv2.destroyAllWindows()
    export_weights(model, "trained_cppn.glsl")

if __name__ == "__main__":
    main()

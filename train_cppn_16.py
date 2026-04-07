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


class GLSL_CPPN(nn.Module):
    """
    1. PyTorch replica of the GLSL Architecture (16 hidden layers)
    """
    def __init__(self):
        super().__init__()
        # --- Level 1 ---
        self.l_buf0 = nn.Linear(8, 4)
        self.l_buf1 = nn.Linear(8, 4)
        self.l_buf2 = nn.Linear(8, 4)
        self.l_buf3 = nn.Linear(8, 4)

        # --- Level 2 ---
        self.l_buf4 = nn.Linear(16, 4)
        self.l_buf5 = nn.Linear(16, 4)
        self.l_buf6 = nn.Linear(16, 4)
        self.l_buf7 = nn.Linear(16, 4)

        # --- Level 3 ---
        self.l_buf8 = nn.Linear(32, 4)
        self.l_buf9 = nn.Linear(32, 4)
        self.l_buf10 = nn.Linear(32, 4)
        self.l_buf11 = nn.Linear(32, 4)

        # --- Level 4 ---
        self.l_buf12 = nn.Linear(48, 4)
        self.l_buf13 = nn.Linear(48, 4)
        self.l_buf14 = nn.Linear(48, 4)
        self.l_buf15 = nn.Linear(48, 4)

        # --- Output Layer ---
        self.l_out = nn.Linear(64, 4)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                std = 0.5 / math.sqrt(m.in_features) 
                nn.init.normal_(m.weight, mean=0.0, std=std)
                nn.init.zeros_(m.bias)

    def forward(self, input_features):
        buf16_17 = input_features

        # WARNING: If you change the activation function here (e.g. from torch.sigmoid),
        # you MUST also update the corresponding GLSL code in the 'export_weights' string!
        # Level 1
        buf0 = torch.sigmoid(self.l_buf0(buf16_17))
        buf1 = torch.sigmoid(self.l_buf1(buf16_17))
        buf2 = torch.sigmoid(self.l_buf2(buf16_17))
        buf3 = torch.sigmoid(self.l_buf3(buf16_17))

        buf0_3 = torch.cat([buf0, buf1, buf2, buf3], dim=-1)

        # Level 2
        buf4 = torch.sigmoid(self.l_buf4(buf0_3))
        buf5 = torch.sigmoid(self.l_buf5(buf0_3))
        buf6 = torch.sigmoid(self.l_buf6(buf0_3))
        buf7 = torch.sigmoid(self.l_buf7(buf0_3))

        buf0_7 = torch.cat([buf0_3, buf4, buf5, buf6, buf7], dim=-1)

        # Level 3
        buf8 = torch.sigmoid(self.l_buf8(buf0_7))
        buf9 = torch.sigmoid(self.l_buf9(buf0_7))
        buf10 = torch.sigmoid(self.l_buf10(buf0_7))
        buf11 = torch.sigmoid(self.l_buf11(buf0_7))

        buf0_11 = torch.cat([buf0_7, buf8, buf9, buf10, buf11], dim=-1)

        # Level 4
        buf12 = torch.sigmoid(self.l_buf12(buf0_11))
        buf13 = torch.sigmoid(self.l_buf13(buf0_11))
        buf14 = torch.sigmoid(self.l_buf14(buf0_11))
        buf15 = torch.sigmoid(self.l_buf15(buf0_11))

        buf0_15 = torch.cat([buf0_11, buf12, buf13, buf14, buf15], dim=-1)

        # Output
        out = torch.sigmoid(self.l_out(buf0_15))
        return out


class SobelEdgeLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.sobel_x = sobel_x.to(device)
        self.sobel_y = sobel_y.to(device)

    def _edges(self, img_flat, size, channels):
        img = img_flat.view(size, size, channels).permute(2, 0, 1).unsqueeze(1)
        ex = torch.nn.functional.conv2d(img, self.sobel_x, padding=1)
        ey = torch.nn.functional.conv2d(img, self.sobel_y, padding=1)
        return torch.sqrt(ex**2 + ey**2 + 1e-8)

    def forward(self, pred_flat, target_flat, size):
        channels = pred_flat.shape[-1]
        pred_edges = self._edges(pred_flat, size, channels)
        target_edges = self._edges(target_flat, size, channels)
        return torch.nn.functional.mse_loss(pred_edges, target_edges)


def get_grid_and_target(
    image_path, size=(128, 128), seeds=(0.3948, 0.36, 0.14), apply_black_to_alpha=False
):
    img = Image.open(image_path).convert("RGBA")
    img = img.resize(size, Image.Resampling.LANCZOS)
    target = np.array(img, dtype=np.float32) / 255.0

    if apply_black_to_alpha:
        max_rgb = np.max(target[..., :3], axis=-1)
        boosted_alpha = np.clip(max_rgb * 1.5, 0.0, 1.0)
        target[..., 3] = boosted_alpha

    x = np.linspace(-1, 1, size[0])
    y = np.linspace(-1, 1, size[1])
    xx, yy = np.meshgrid(x, y)

    inputs = np.stack(
        [
            xx,
            yy,
            np.full_like(xx, seeds[0]),
            np.full_like(xx, seeds[1]),
            np.full_like(xx, seeds[2]),
            np.sqrt(xx**2 + yy**2),
            np.zeros_like(xx),
            np.zeros_like(xx)
        ], axis=-1
    )

    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(
        target, dtype=torch.float32
    )


def export_weights(model, filename="trained_cppn_16.glsl", seeds=(0.3948, 0.36, 0.14)):
    def format_vec4(vec):
        return f"vec4({vec[0]:.6f},{vec[1]:.6f},{vec[2]:.6f},{vec[3]:.6f})"

    def format_mat4(mat):
        c0, c1, c2, c3 = [format_vec4(mat[:, i]) for i in range(4)]
        return f"mat4({c0},{c1},{c2},{c3})"

    def get_layer(layer):
        return layer.weight.detach().cpu().numpy(), layer.bias.detach().cpu().numpy()

    kwargs = {"seed0": seeds[0], "seed1": seeds[1], "seed2": seeds[2]}
    layers = [
        model.l_buf0, model.l_buf1, model.l_buf2, model.l_buf3,
        model.l_buf4, model.l_buf5, model.l_buf6, model.l_buf7,
        model.l_buf8, model.l_buf9, model.l_buf10, model.l_buf11,
        model.l_buf12, model.l_buf13, model.l_buf14, model.l_buf15,
        model.l_out
    ]

    for i, layer in enumerate(layers, start=1):
        W, b = get_layer(layer)
        kwargs[f"b{i}"] = format_vec4(b)
        
        num_mat4 = W.shape[1] // 4
        for c in range(num_mat4):
            kwargs[f"w{i}_{c+1}"] = format_mat4(W[:, c*4 : (c+1)*4])

    glsl_template = """#version 120
#ifdef GL_ES
precision highp float;
#endif

varying vec2 v_TexCoord;
uniform float g_Time;
uniform vec3 g_Screen;
uniform float uContrast; // {{"material":"Contrast","default":1,"range":[0,2]}}
uniform float uBrightness; // {{"material":"Brightness","default":0,"range":[-1,1]}}
uniform float uNoiseSpeed0; // {{"material":"uNoiseSpeed0","default":0.3,"range":[0,2]}}
uniform float uNoiseSpeed1; // {{"material":"uNoiseSpeed1","default":0.69,"range":[0,2]}}
uniform float uNoiseSpeed2; // {{"material":"uNoiseSpeed2","default":0.44,"range":[0,2]}}
uniform float uHueShift; // {{"material":"uHueShift","default":1,"range":[0,600]}}
uniform float uNoise; // {{"material":"uNoise","default":0.1,"range":[0,1]}}
uniform float uScan; // {{"material":"uScan","default":0,"range":[0,1]}}
uniform float uScanFreq; // {{"material":"uScanFreq","default":0,"range":[0,1]}}
uniform float uWarp; // {{"material":"uWarp","default":0.2,"range":[0,1]}}
uniform float uRotation;  // {{"material":"Rotation","default":0,"range":[0 , 360]}}
#define uTime g_Time
#define uResolution g_Screen

const float PI = 3.141592653589793;
const float EPSILON = 0.001;
const float E = 2.71828182845904523536;
const float HALF = 0.5;

vec4 mmul(mat4 m, vec4 v) {{
    return v.x*m[0] + v.y*m[1] + v.z*m[2] + v.w*m[3];
}}
vec3 mmul(mat3 m, vec3 v) {{
    return v.x*m[0] + v.y*m[1] + v.z*m[2];
}}

vec2 rotate(vec2 v, float angle) {{
    float s = sin(angle);
    float c = cos(angle);
    return vec2(v.x * c - v.y * s, v.x * s + v.y * c);
}}

float noise(vec2 coord) {{
    float G = E;
    vec2 r = (G * sin(G * coord));
    return fract(r.x * r.y * (1.0 + coord.x));
}}

float hash(vec2 p) {{
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}}

float hermiteNoise(vec2 uv) {{
    vec2 i = floor(uv);
    vec2 f = fract(uv);

    vec2 u = f * f * (3.0 - 2.0 * f);

    float a = hash(i + vec2(0.0, 0.0));
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    return mix(mix(a, b, u.x), 
               mix(c, d, u.x), u.y);
}}

const mat3 rgb2yiq=mat3(0.299,0.587,0.114,0.596,-0.274,-0.322,0.211,-0.523,0.312);
const mat3 yiq2rgb=mat3(1.0,0.956,0.621,1.0,-0.272,-0.647,1.0,-1.106,1.703);

vec3 hueShiftRGB(vec3 col,float deg){{
    vec3 yiq=mmul(rgb2yiq, col);
    float rad=radians(deg);
    float ch=cos(rad),sh=sin(rad);
    vec3 yiqShift=vec3(yiq.x,yiq.y*ch-yiq.z*sh,yiq.y*sh+yiq.z*ch);
    return clamp(mmul(yiq2rgb, yiqShift),0.0,1.0);
}}

vec4 adjustBrightnessContrast(vec4 color, float brightness, float contrast) {{
    vec3 adjustedRgb = (color.rgb - 0.5) * contrast + 0.5 + brightness;
    return vec4(adjustedRgb, color.a);
}}

vec4 sigmoid(vec4 x){{return 1./(1.+exp(-x));}}

vec4 cppn_fn(vec2 coordinate, float in0, float in1, float in2) {{
    // --- INPUT LAYER ---
    vec4 in_buf[18];
    in_buf[16] = vec4(coordinate.x, coordinate.y, {seed0:.6f} + in0, {seed1:.6f} + in1);
    in_buf[17] = vec4({seed2:.6f} + in2, sqrt(coordinate.x * coordinate.x + coordinate.y * coordinate.y), 0., 0.);
    
    // --- WEIGHTS & BIASES ---
    // Level 1
    mat4 w1_1 = {w1_1}; mat4 w1_2 = {w1_2}; vec4 b1 = {b1};
    mat4 w2_1 = {w2_1}; mat4 w2_2 = {w2_2}; vec4 b2 = {b2};
    mat4 w3_1 = {w3_1}; mat4 w3_2 = {w3_2}; vec4 b3 = {b3};
    mat4 w4_1 = {w4_1}; mat4 w4_2 = {w4_2}; vec4 b4 = {b4};

    // Level 2
    mat4 w5_1 = {w5_1}; mat4 w5_2 = {w5_2}; mat4 w5_3 = {w5_3}; mat4 w5_4 = {w5_4}; vec4 b5 = {b5};
    mat4 w6_1 = {w6_1}; mat4 w6_2 = {w6_2}; mat4 w6_3 = {w6_3}; mat4 w6_4 = {w6_4}; vec4 b6 = {b6};
    mat4 w7_1 = {w7_1}; mat4 w7_2 = {w7_2}; mat4 w7_3 = {w7_3}; mat4 w7_4 = {w7_4}; vec4 b7 = {b7};
    mat4 w8_1 = {w8_1}; mat4 w8_2 = {w8_2}; mat4 w8_3 = {w8_3}; mat4 w8_4 = {w8_4}; vec4 b8 = {b8};

    // Level 3
    mat4 w9_1 = {w9_1}; mat4 w9_2 = {w9_2}; mat4 w9_3 = {w9_3}; mat4 w9_4 = {w9_4}; mat4 w9_5 = {w9_5}; mat4 w9_6 = {w9_6}; mat4 w9_7 = {w9_7}; mat4 w9_8 = {w9_8}; vec4 b9 = {b9};
    mat4 w10_1 = {w10_1}; mat4 w10_2 = {w10_2}; mat4 w10_3 = {w10_3}; mat4 w10_4 = {w10_4}; mat4 w10_5 = {w10_5}; mat4 w10_6 = {w10_6}; mat4 w10_7 = {w10_7}; mat4 w10_8 = {w10_8}; vec4 b10 = {b10};
    mat4 w11_1 = {w11_1}; mat4 w11_2 = {w11_2}; mat4 w11_3 = {w11_3}; mat4 w11_4 = {w11_4}; mat4 w11_5 = {w11_5}; mat4 w11_6 = {w11_6}; mat4 w11_7 = {w11_7}; mat4 w11_8 = {w11_8}; vec4 b11 = {b11};
    mat4 w12_1 = {w12_1}; mat4 w12_2 = {w12_2}; mat4 w12_3 = {w12_3}; mat4 w12_4 = {w12_4}; mat4 w12_5 = {w12_5}; mat4 w12_6 = {w12_6}; mat4 w12_7 = {w12_7}; mat4 w12_8 = {w12_8}; vec4 b12 = {b12};

    // Level 4
    mat4 w13_1 = {w13_1}; mat4 w13_2 = {w13_2}; mat4 w13_3 = {w13_3}; mat4 w13_4 = {w13_4}; mat4 w13_5 = {w13_5}; mat4 w13_6 = {w13_6}; mat4 w13_7 = {w13_7}; mat4 w13_8 = {w13_8}; mat4 w13_9 = {w13_9}; mat4 w13_10 = {w13_10}; mat4 w13_11 = {w13_11}; mat4 w13_12 = {w13_12}; vec4 b13 = {b13};
    mat4 w14_1 = {w14_1}; mat4 w14_2 = {w14_2}; mat4 w14_3 = {w14_3}; mat4 w14_4 = {w14_4}; mat4 w14_5 = {w14_5}; mat4 w14_6 = {w14_6}; mat4 w14_7 = {w14_7}; mat4 w14_8 = {w14_8}; mat4 w14_9 = {w14_9}; mat4 w14_10 = {w14_10}; mat4 w14_11 = {w14_11}; mat4 w14_12 = {w14_12}; vec4 b14 = {b14};
    mat4 w15_1 = {w15_1}; mat4 w15_2 = {w15_2}; mat4 w15_3 = {w15_3}; mat4 w15_4 = {w15_4}; mat4 w15_5 = {w15_5}; mat4 w15_6 = {w15_6}; mat4 w15_7 = {w15_7}; mat4 w15_8 = {w15_8}; mat4 w15_9 = {w15_9}; mat4 w15_10 = {w15_10}; mat4 w15_11 = {w15_11}; mat4 w15_12 = {w15_12}; vec4 b15 = {b15};
    mat4 w16_1 = {w16_1}; mat4 w16_2 = {w16_2}; mat4 w16_3 = {w16_3}; mat4 w16_4 = {w16_4}; mat4 w16_5 = {w16_5}; mat4 w16_6 = {w16_6}; mat4 w16_7 = {w16_7}; mat4 w16_8 = {w16_8}; mat4 w16_9 = {w16_9}; mat4 w16_10 = {w16_10}; mat4 w16_11 = {w16_11}; mat4 w16_12 = {w16_12}; vec4 b16 = {b16};

    // Output Layer
    mat4 w17_1 = {w17_1}; mat4 w17_2 = {w17_2}; mat4 w17_3 = {w17_3}; mat4 w17_4 = {w17_4}; mat4 w17_5 = {w17_5}; mat4 w17_6 = {w17_6}; mat4 w17_7 = {w17_7}; mat4 w17_8 = {w17_8}; mat4 w17_9 = {w17_9}; mat4 w17_10 = {w17_10}; mat4 w17_11 = {w17_11}; mat4 w17_12 = {w17_12}; mat4 w17_13 = {w17_13}; mat4 w17_14 = {w17_14}; mat4 w17_15 = {w17_15}; mat4 w17_16 = {w17_16}; vec4 b17 = {b17};

    // --- APPLYING NEURAL NETWORK ---
    // WARNING: If you changed the activation function in the PyTorch 'forward' method,
    // you MUST also update the mathematical equivalents below!
    
    // Level 1 calculation
    in_buf[0] = mmul(w1_1, in_buf[16]) + mmul(w1_2, in_buf[17]) + b1;
    in_buf[1] = mmul(w2_1, in_buf[16]) + mmul(w2_2, in_buf[17]) + b2;
    in_buf[2] = mmul(w3_1, in_buf[16]) + mmul(w3_2, in_buf[17]) + b3;
    in_buf[3] = mmul(w4_1, in_buf[16]) + mmul(w4_2, in_buf[17]) + b4;
    in_buf[0] = sigmoid(in_buf[0]); in_buf[1] = sigmoid(in_buf[1]);
    in_buf[2] = sigmoid(in_buf[2]); in_buf[3] = sigmoid(in_buf[3]);

    // Level 2 calculation
    in_buf[4] = mmul(w5_1, in_buf[0]) + mmul(w5_2, in_buf[1]) + mmul(w5_3, in_buf[2]) + mmul(w5_4, in_buf[3]) + b5;
    in_buf[5] = mmul(w6_1, in_buf[0]) + mmul(w6_2, in_buf[1]) + mmul(w6_3, in_buf[2]) + mmul(w6_4, in_buf[3]) + b6;
    in_buf[6] = mmul(w7_1, in_buf[0]) + mmul(w7_2, in_buf[1]) + mmul(w7_3, in_buf[2]) + mmul(w7_4, in_buf[3]) + b7;
    in_buf[7] = mmul(w8_1, in_buf[0]) + mmul(w8_2, in_buf[1]) + mmul(w8_3, in_buf[2]) + mmul(w8_4, in_buf[3]) + b8;
    in_buf[4] = sigmoid(in_buf[4]); in_buf[5] = sigmoid(in_buf[5]);
    in_buf[6] = sigmoid(in_buf[6]); in_buf[7] = sigmoid(in_buf[7]);
    
    // Level 3 calculation
    in_buf[8]  = mmul(w9_1, in_buf[0]) + mmul(w9_2, in_buf[1]) + mmul(w9_3, in_buf[2]) + mmul(w9_4, in_buf[3]) + mmul(w9_5, in_buf[4]) + mmul(w9_6, in_buf[5]) + mmul(w9_7, in_buf[6]) + mmul(w9_8, in_buf[7]) + b9;
    in_buf[9]  = mmul(w10_1, in_buf[0]) + mmul(w10_2, in_buf[1]) + mmul(w10_3, in_buf[2]) + mmul(w10_4, in_buf[3]) + mmul(w10_5, in_buf[4]) + mmul(w10_6, in_buf[5]) + mmul(w10_7, in_buf[6]) + mmul(w10_8, in_buf[7]) + b10;
    in_buf[10] = mmul(w11_1, in_buf[0]) + mmul(w11_2, in_buf[1]) + mmul(w11_3, in_buf[2]) + mmul(w11_4, in_buf[3]) + mmul(w11_5, in_buf[4]) + mmul(w11_6, in_buf[5]) + mmul(w11_7, in_buf[6]) + mmul(w11_8, in_buf[7]) + b11;
    in_buf[11] = mmul(w12_1, in_buf[0]) + mmul(w12_2, in_buf[1]) + mmul(w12_3, in_buf[2]) + mmul(w12_4, in_buf[3]) + mmul(w12_5, in_buf[4]) + mmul(w12_6, in_buf[5]) + mmul(w12_7, in_buf[6]) + mmul(w12_8, in_buf[7]) + b12;
    in_buf[8] = sigmoid(in_buf[8]); in_buf[9]  = sigmoid(in_buf[9]);
    in_buf[10] = sigmoid(in_buf[10]); in_buf[11] = sigmoid(in_buf[11]);

    // Level 4 calculation
    in_buf[12] = mmul(w13_1, in_buf[0]) + mmul(w13_2, in_buf[1]) + mmul(w13_3, in_buf[2]) + mmul(w13_4, in_buf[3]) + mmul(w13_5, in_buf[4]) + mmul(w13_6, in_buf[5]) + mmul(w13_7, in_buf[6]) + mmul(w13_8, in_buf[7]) + mmul(w13_9, in_buf[8]) + mmul(w13_10, in_buf[9]) + mmul(w13_11, in_buf[10]) + mmul(w13_12, in_buf[11]) + b13;
    in_buf[13] = mmul(w14_1, in_buf[0]) + mmul(w14_2, in_buf[1]) + mmul(w14_3, in_buf[2]) + mmul(w14_4, in_buf[3]) + mmul(w14_5, in_buf[4]) + mmul(w14_6, in_buf[5]) + mmul(w14_7, in_buf[6]) + mmul(w14_8, in_buf[7]) + mmul(w14_9, in_buf[8]) + mmul(w14_10, in_buf[9]) + mmul(w14_11, in_buf[10]) + mmul(w14_12, in_buf[11]) + b14;
    in_buf[14] = mmul(w15_1, in_buf[0]) + mmul(w15_2, in_buf[1]) + mmul(w15_3, in_buf[2]) + mmul(w15_4, in_buf[3]) + mmul(w15_5, in_buf[4]) + mmul(w15_6, in_buf[5]) + mmul(w15_7, in_buf[6]) + mmul(w15_8, in_buf[7]) + mmul(w15_9, in_buf[8]) + mmul(w15_10, in_buf[9]) + mmul(w15_11, in_buf[10]) + mmul(w15_12, in_buf[11]) + b15;
    in_buf[15] = mmul(w16_1, in_buf[0]) + mmul(w16_2, in_buf[1]) + mmul(w16_3, in_buf[2]) + mmul(w16_4, in_buf[3]) + mmul(w16_5, in_buf[4]) + mmul(w16_6, in_buf[5]) + mmul(w16_7, in_buf[6]) + mmul(w16_8, in_buf[7]) + mmul(w16_9, in_buf[8]) + mmul(w16_10, in_buf[9]) + mmul(w16_11, in_buf[10]) + mmul(w16_12, in_buf[11]) + b16;
    in_buf[12] = sigmoid(in_buf[12]); in_buf[13] = sigmoid(in_buf[13]);
    in_buf[14] = sigmoid(in_buf[14]); in_buf[15] = sigmoid(in_buf[15]);

    // Output Layer calculation
    in_buf[0] = mmul(w17_1, in_buf[0]) + mmul(w17_2, in_buf[1]) + mmul(w17_3, in_buf[2]) + mmul(w17_4, in_buf[3]) + mmul(w17_5, in_buf[4]) + mmul(w17_6, in_buf[5]) + mmul(w17_7, in_buf[6]) + mmul(w17_8, in_buf[7]) + mmul(w17_9, in_buf[8]) + mmul(w17_10, in_buf[9]) + mmul(w17_11, in_buf[10]) + mmul(w17_12, in_buf[11]) + mmul(w17_13, in_buf[12]) + mmul(w17_14, in_buf[13]) + mmul(w17_15, in_buf[14]) + mmul(w17_16, in_buf[15]) + b17;
    in_buf[0] = sigmoid(in_buf[0]);
    
    return vec4(in_buf[0].x, in_buf[0].y, in_buf[0].z, in_buf[0].w);
}}

void preProcess(out vec4 fragColor,in vec2 fragCoord){{
    vec2 uv=fragCoord/uResolution.xy*2.-1.;
    uv.y*=-1.;
    float rotAngle = uRotation * PI / 180.0;
    uv = rotate(uv, rotAngle);
    uv+=uWarp*hermiteNoise(vec2(sin(uv.y*6.283+uTime*0.5),cos(uv.x*6.283+uTime*0.5)))*0.05;
    fragColor=cppn_fn(uv,0.1*hermiteNoise(uv+uNoiseSpeed0*uTime),0.1*hermiteNoise(uv+uNoiseSpeed1*uTime),0.1*hermiteNoise(uv+uNoiseSpeed2*uTime));
}}

void main(){{
    vec2 fragCoord = v_TexCoord.xy * uResolution.xy;
    
    vec4 col;
    preProcess(col, fragCoord);

    col.rgb=hueShiftRGB(col.rgb,uHueShift);
    col = adjustBrightnessContrast(col,uBrightness, uContrast);
    
    float scanline_val=sin(fragCoord.y*uScanFreq)*0.5+0.5;
    col.rgb*=1.-(scanline_val*scanline_val)*uScan;
    float rnd = noise(fragCoord.xy);
    col.rgb-= rnd / 15.0 *uNoise;
    gl_FragColor=clamp(col,0.0,1.0);
}}"""

    code = glsl_template.format(**kwargs)

    with open(filename, "w") as f:
        f.write(code)
    print(f"\nSaved trained GLSL to: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a CPPN to match a picture and export to GLSL"
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to target image (optional, will open file dialog if omitted)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Training resolution (larger = more GPU work, default: 256)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.002, help="Learning rate (default: 0.002)"
    )
    parser.add_argument(
        "--display-every",
        type=int,
        default=100,
        help="Update preview every N steps (higher = faster training, default: 100)",
    )
    parser.add_argument(
        "--input-noise",
        type=float,
        default=0.0,
        help="Std of Gaussian noise added to inputs each step to prevent stagnation (default: 0.01, set 0 to disable)",
    )
    parser.add_argument(
        "--perturb-scale",
        type=float,
        default=0.0,
        help="Std of weight perturbation applied when LR scheduler reduces LR (default: 0.02, set 0 to disable)",
    )
    parser.add_argument(
        "--edge-weight",
        type=float,
        default=0,
        help="Weight for Sobel edge sharpness loss (default: 0.3, higher = sharper, 0 = disable)",
    )
    parser.add_argument(
        "--black-to-alpha",
        action="store_true",
        help="Set alpha based on RGB brightness (distance from black = alpha).",
    )
    args = parser.parse_args()

    image_path = args.image

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
                ("All files", "*.*"),
            ],
        )
        if not image_path:
            print("No image selected. Exiting.")
            sys.exit(0)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif HAS_DIRECTML and torch_directml.is_available():
        device = torch_directml.device()
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print(f"Loading image {image_path} at {args.size}x{args.size}...")

    import random

    seeds = (
        round(random.uniform(0.01, 0.99), 4),
        round(random.uniform(0.01, 0.99), 4),
        round(random.uniform(0.01, 0.99), 4),
    )
    print(f"Noise seeds this run: {seeds}")

    inputs, targets = get_grid_and_target(
        image_path,
        size=(args.size, args.size),
        seeds=seeds,
        apply_black_to_alpha=args.black_to_alpha,
    )

    inputs_flat = inputs.view(-1, 8).to(device)
    targets_flat = targets.view(-1, 4).to(device)

    model = GLSL_CPPN().to(device)

    is_dml = False
    try:
        if "privateuseone" in str(device):
            is_dml = True
    except:
        pass

    if not is_dml:
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception:
            pass 

    if HAS_DIRECTML and is_dml:
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9)  # type: ignore
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)  # type: ignore

    min_lr = 1e-5

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50, min_lr=min_lr
    )

    criterion = nn.MSELoss()
    edge_criterion = SobelEdgeLoss(device) if args.edge_weight > 0 else None
    edge_weight = args.edge_weight

    print("\nStarting Training! Focus the image window and press 'q' to stop & save.")

    display_every = args.display_every
    display_scale = 2 if args.size >= 512 else 4

    cv2.namedWindow("CPPN Trainer")

    checker = np.zeros((args.size, args.size, 3), dtype=np.float32)
    for cy in range(0, args.size, 16):
        for cx in range(0, args.size, 16):
            val = 0.8 if ((cx // 16) + (cy // 16)) % 2 == 0 else 0.4
            checker[cy : cy + 16, cx : cx + 16] = val

    def composite_on_checkerboard(img_data):
        alpha = img_data[:, :, 3:4]
        rgb = img_data[:, :, :3] * alpha + checker * (1.0 - alpha)
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    step = 0
    prev_lr = optimizer.param_groups[0]["lr"]
    try:
        while True:
            optimizer.zero_grad()

            if args.input_noise > 0 and step % 1001 == 0:
                noisy_inputs = (
                    inputs_flat + torch.randn_like(inputs_flat) * args.input_noise
                )
                pred_flat = model(noisy_inputs)
            else:
                pred_flat = model(inputs_flat)

            mse_loss = criterion(pred_flat[:, :4], targets_flat)
            if edge_criterion is not None:
                edge_loss = edge_criterion(pred_flat[:, :4], targets_flat, args.size)
                loss = mse_loss + edge_weight * edge_loss
            else:
                loss = mse_loss
            loss.backward()
            optimizer.step()

            if step % display_every == 0:
                if step >= 5000:
                    scheduler.step(loss.item())

                current_lr = optimizer.param_groups[0]["lr"]

                if current_lr <= (min_lr + 1e-9):
                    print(
                        f"\n\nTraining completed: Learning rate has decayed to {current_lr:.6g}. Saving and exiting."
                    )
                    break

                if args.perturb_scale > 0 and current_lr < prev_lr:
                    with torch.no_grad():
                        for param in model.parameters():
                            param.add_(torch.randn_like(param) * args.perturb_scale)
                    print(
                        f"  [step {step}] LR reduced to {current_lr:.6f} — weight perturbation applied"
                    )
                prev_lr = current_lr

                if cv2.getWindowProperty("CPPN Trainer", cv2.WND_PROP_VISIBLE) < 1:
                    print("\n\nTraining stopped by user (Window closed).")
                    break

                with torch.no_grad():
                    pred_img = (
                        pred_flat.detach().view(args.size, args.size, 4).cpu().numpy()
                    )
                    target_img = targets.numpy()

                    pred_bgr = composite_on_checkerboard(pred_img)
                    target_bgr = composite_on_checkerboard(target_img)

                    display_bgr = np.hstack((target_bgr, pred_bgr))
                    display_bgr = cv2.resize(
                        display_bgr,
                        (args.size * display_scale, args.size * (display_scale // 2)),
                        interpolation=cv2.INTER_NEAREST,
                    )

                    panel_height = 80
                    text_panel = (
                        np.ones((panel_height, display_bgr.shape[1], 3), dtype=np.uint8)
                        * 255
                    )
                    display_bgr = np.vstack((display_bgr, text_panel))

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    status_text = f"Step: {step:05d} | Loss: {loss.item():.5f} | LR: {current_lr:.5f}"
                    base_y = args.size * (display_scale // 2)

                    cv2.putText(
                        display_bgr,
                        status_text,
                        (10, base_y + 30),
                        font,
                        0.5,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        display_bgr,
                        "Press 'Q' to save & exit",
                        (10, base_y + 60),
                        font,
                        0.5,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

                    cv2.imshow("CPPN Trainer", display_bgr)

                    key = cv2.waitKey(1)
                    if key & 0xFF == ord("q"):
                        print("\n\nTraining stopped by user.")
                        break
            step += 1

    except KeyboardInterrupt:
        print("\n\nTraining stopped with Ctrl+C")

    cv2.destroyAllWindows()

    export_weights(model, "trained_cppn_16.glsl", seeds=seeds)


if __name__ == "__main__":
    main()

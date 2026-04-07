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
    1. PyTorch replica of the GLSL Architecture (32 hidden layers)
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

        # --- Level 5 ---
        self.l_buf16 = nn.Linear(64, 4)
        self.l_buf17 = nn.Linear(64, 4)
        self.l_buf18 = nn.Linear(64, 4)
        self.l_buf19 = nn.Linear(64, 4)

        # --- Level 6 ---
        self.l_buf20 = nn.Linear(80, 4)
        self.l_buf21 = nn.Linear(80, 4)
        self.l_buf22 = nn.Linear(80, 4)
        self.l_buf23 = nn.Linear(80, 4)

        # --- Level 7 ---
        self.l_buf24 = nn.Linear(96, 4)
        self.l_buf25 = nn.Linear(96, 4)
        self.l_buf26 = nn.Linear(96, 4)
        self.l_buf27 = nn.Linear(96, 4)

        # --- Level 8 ---
        self.l_buf28 = nn.Linear(112, 4)
        self.l_buf29 = nn.Linear(112, 4)
        self.l_buf30 = nn.Linear(112, 4)
        self.l_buf31 = nn.Linear(112, 4)

        # --- Output Layer ---
        self.l_out = nn.Linear(128, 4)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                std = 0.5 / math.sqrt(m.in_features) 
                nn.init.normal_(m.weight, mean=0.0, std=std)
                nn.init.zeros_(m.bias)

    def forward(self, input_features):
        buf32_33 = input_features

        # Level 1
        buf0 = torch.sigmoid(self.l_buf0(buf32_33))
        buf1 = torch.sigmoid(self.l_buf1(buf32_33))
        buf2 = torch.sigmoid(self.l_buf2(buf32_33))
        buf3 = torch.sigmoid(self.l_buf3(buf32_33))

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

        # Level 5
        buf16 = torch.sigmoid(self.l_buf16(buf0_15))
        buf17 = torch.sigmoid(self.l_buf17(buf0_15))
        buf18 = torch.sigmoid(self.l_buf18(buf0_15))
        buf19 = torch.sigmoid(self.l_buf19(buf0_15))

        buf0_19 = torch.cat([buf0_15, buf16, buf17, buf18, buf19], dim=-1)

        # Level 6
        buf20 = torch.sigmoid(self.l_buf20(buf0_19))
        buf21 = torch.sigmoid(self.l_buf21(buf0_19))
        buf22 = torch.sigmoid(self.l_buf22(buf0_19))
        buf23 = torch.sigmoid(self.l_buf23(buf0_19))

        buf0_23 = torch.cat([buf0_19, buf20, buf21, buf22, buf23], dim=-1)

        # Level 7
        buf24 = torch.sigmoid(self.l_buf24(buf0_23))
        buf25 = torch.sigmoid(self.l_buf25(buf0_23))
        buf26 = torch.sigmoid(self.l_buf26(buf0_23))
        buf27 = torch.sigmoid(self.l_buf27(buf0_23))

        buf0_27 = torch.cat([buf0_23, buf24, buf25, buf26, buf27], dim=-1)

        # Level 8
        buf28 = torch.sigmoid(self.l_buf28(buf0_27))
        buf29 = torch.sigmoid(self.l_buf29(buf0_27))
        buf30 = torch.sigmoid(self.l_buf30(buf0_27))
        buf31 = torch.sigmoid(self.l_buf31(buf0_27))

        buf0_31 = torch.cat([buf0_27, buf28, buf29, buf30, buf31], dim=-1)

        # Output
        out = torch.sigmoid(self.l_out(buf0_31))
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

    # PREVENT SIGMOID DEATH: Absolute 0.0 (black) or 1.0 (white) forces the network 
    # to push its weights to negative or positive infinity, killing all gradients and causing
    # colors to collapse. Clamping it slightly avoids this infinite-weight trap.
    target = np.clip(target, 0.02, 0.98)

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


def export_weights(model, filename="trained_cppn_32.glsl", seeds=(0.3948, 0.36, 0.14)):
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
        model.l_buf16, model.l_buf17, model.l_buf18, model.l_buf19,
        model.l_buf20, model.l_buf21, model.l_buf22, model.l_buf23,
        model.l_buf24, model.l_buf25, model.l_buf26, model.l_buf27,
        model.l_buf28, model.l_buf29, model.l_buf30, model.l_buf31,
        model.l_out,
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
    vec4 in_buf[34];
    in_buf[32] = vec4(coordinate.x, coordinate.y, {seed0:.6f} + in0, {seed1:.6f} + in1);
    in_buf[33] = vec4({seed2:.6f} + in2, sqrt(coordinate.x * coordinate.x + coordinate.y * coordinate.y), 0., 0.);
    
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

    // Level 5
    mat4 w17_1 = {w17_1}; mat4 w17_2 = {w17_2}; mat4 w17_3 = {w17_3}; mat4 w17_4 = {w17_4}; mat4 w17_5 = {w17_5}; mat4 w17_6 = {w17_6}; mat4 w17_7 = {w17_7}; mat4 w17_8 = {w17_8}; mat4 w17_9 = {w17_9}; mat4 w17_10 = {w17_10}; mat4 w17_11 = {w17_11}; mat4 w17_12 = {w17_12}; mat4 w17_13 = {w17_13}; mat4 w17_14 = {w17_14}; mat4 w17_15 = {w17_15}; mat4 w17_16 = {w17_16}; vec4 b17 = {b17};
    mat4 w18_1 = {w18_1}; mat4 w18_2 = {w18_2}; mat4 w18_3 = {w18_3}; mat4 w18_4 = {w18_4}; mat4 w18_5 = {w18_5}; mat4 w18_6 = {w18_6}; mat4 w18_7 = {w18_7}; mat4 w18_8 = {w18_8}; mat4 w18_9 = {w18_9}; mat4 w18_10 = {w18_10}; mat4 w18_11 = {w18_11}; mat4 w18_12 = {w18_12}; mat4 w18_13 = {w18_13}; mat4 w18_14 = {w18_14}; mat4 w18_15 = {w18_15}; mat4 w18_16 = {w18_16}; vec4 b18 = {b18};
    mat4 w19_1 = {w19_1}; mat4 w19_2 = {w19_2}; mat4 w19_3 = {w19_3}; mat4 w19_4 = {w19_4}; mat4 w19_5 = {w19_5}; mat4 w19_6 = {w19_6}; mat4 w19_7 = {w19_7}; mat4 w19_8 = {w19_8}; mat4 w19_9 = {w19_9}; mat4 w19_10 = {w19_10}; mat4 w19_11 = {w19_11}; mat4 w19_12 = {w19_12}; mat4 w19_13 = {w19_13}; mat4 w19_14 = {w19_14}; mat4 w19_15 = {w19_15}; mat4 w19_16 = {w19_16}; vec4 b19 = {b19};
    mat4 w20_1 = {w20_1}; mat4 w20_2 = {w20_2}; mat4 w20_3 = {w20_3}; mat4 w20_4 = {w20_4}; mat4 w20_5 = {w20_5}; mat4 w20_6 = {w20_6}; mat4 w20_7 = {w20_7}; mat4 w20_8 = {w20_8}; mat4 w20_9 = {w20_9}; mat4 w20_10 = {w20_10}; mat4 w20_11 = {w20_11}; mat4 w20_12 = {w20_12}; mat4 w20_13 = {w20_13}; mat4 w20_14 = {w20_14}; mat4 w20_15 = {w20_15}; mat4 w20_16 = {w20_16}; vec4 b20 = {b20};

    // Level 6
    mat4 w21_1 = {w21_1}; mat4 w21_2 = {w21_2}; mat4 w21_3 = {w21_3}; mat4 w21_4 = {w21_4}; mat4 w21_5 = {w21_5}; mat4 w21_6 = {w21_6}; mat4 w21_7 = {w21_7}; mat4 w21_8 = {w21_8}; mat4 w21_9 = {w21_9}; mat4 w21_10 = {w21_10}; mat4 w21_11 = {w21_11}; mat4 w21_12 = {w21_12}; mat4 w21_13 = {w21_13}; mat4 w21_14 = {w21_14}; mat4 w21_15 = {w21_15}; mat4 w21_16 = {w21_16}; mat4 w21_17 = {w21_17}; mat4 w21_18 = {w21_18}; mat4 w21_19 = {w21_19}; mat4 w21_20 = {w21_20}; vec4 b21 = {b21};
    mat4 w22_1 = {w22_1}; mat4 w22_2 = {w22_2}; mat4 w22_3 = {w22_3}; mat4 w22_4 = {w22_4}; mat4 w22_5 = {w22_5}; mat4 w22_6 = {w22_6}; mat4 w22_7 = {w22_7}; mat4 w22_8 = {w22_8}; mat4 w22_9 = {w22_9}; mat4 w22_10 = {w22_10}; mat4 w22_11 = {w22_11}; mat4 w22_12 = {w22_12}; mat4 w22_13 = {w22_13}; mat4 w22_14 = {w22_14}; mat4 w22_15 = {w22_15}; mat4 w22_16 = {w22_16}; mat4 w22_17 = {w22_17}; mat4 w22_18 = {w22_18}; mat4 w22_19 = {w22_19}; mat4 w22_20 = {w22_20}; vec4 b22 = {b22};
    mat4 w23_1 = {w23_1}; mat4 w23_2 = {w23_2}; mat4 w23_3 = {w23_3}; mat4 w23_4 = {w23_4}; mat4 w23_5 = {w23_5}; mat4 w23_6 = {w23_6}; mat4 w23_7 = {w23_7}; mat4 w23_8 = {w23_8}; mat4 w23_9 = {w23_9}; mat4 w23_10 = {w23_10}; mat4 w23_11 = {w23_11}; mat4 w23_12 = {w23_12}; mat4 w23_13 = {w23_13}; mat4 w23_14 = {w23_14}; mat4 w23_15 = {w23_15}; mat4 w23_16 = {w23_16}; mat4 w23_17 = {w23_17}; mat4 w23_18 = {w23_18}; mat4 w23_19 = {w23_19}; mat4 w23_20 = {w23_20}; vec4 b23 = {b23};
    mat4 w24_1 = {w24_1}; mat4 w24_2 = {w24_2}; mat4 w24_3 = {w24_3}; mat4 w24_4 = {w24_4}; mat4 w24_5 = {w24_5}; mat4 w24_6 = {w24_6}; mat4 w24_7 = {w24_7}; mat4 w24_8 = {w24_8}; mat4 w24_9 = {w24_9}; mat4 w24_10 = {w24_10}; mat4 w24_11 = {w24_11}; mat4 w24_12 = {w24_12}; mat4 w24_13 = {w24_13}; mat4 w24_14 = {w24_14}; mat4 w24_15 = {w24_15}; mat4 w24_16 = {w24_16}; mat4 w24_17 = {w24_17}; mat4 w24_18 = {w24_18}; mat4 w24_19 = {w24_19}; mat4 w24_20 = {w24_20}; vec4 b24 = {b24};

    // Level 7
    mat4 w25_1 = {w25_1}; mat4 w25_2 = {w25_2}; mat4 w25_3 = {w25_3}; mat4 w25_4 = {w25_4}; mat4 w25_5 = {w25_5}; mat4 w25_6 = {w25_6}; mat4 w25_7 = {w25_7}; mat4 w25_8 = {w25_8}; mat4 w25_9 = {w25_9}; mat4 w25_10 = {w25_10}; mat4 w25_11 = {w25_11}; mat4 w25_12 = {w25_12}; mat4 w25_13 = {w25_13}; mat4 w25_14 = {w25_14}; mat4 w25_15 = {w25_15}; mat4 w25_16 = {w25_16}; mat4 w25_17 = {w25_17}; mat4 w25_18 = {w25_18}; mat4 w25_19 = {w25_19}; mat4 w25_20 = {w25_20}; mat4 w25_21 = {w25_21}; mat4 w25_22 = {w25_22}; mat4 w25_23 = {w25_23}; mat4 w25_24 = {w25_24}; vec4 b25 = {b25};
    mat4 w26_1 = {w26_1}; mat4 w26_2 = {w26_2}; mat4 w26_3 = {w26_3}; mat4 w26_4 = {w26_4}; mat4 w26_5 = {w26_5}; mat4 w26_6 = {w26_6}; mat4 w26_7 = {w26_7}; mat4 w26_8 = {w26_8}; mat4 w26_9 = {w26_9}; mat4 w26_10 = {w26_10}; mat4 w26_11 = {w26_11}; mat4 w26_12 = {w26_12}; mat4 w26_13 = {w26_13}; mat4 w26_14 = {w26_14}; mat4 w26_15 = {w26_15}; mat4 w26_16 = {w26_16}; mat4 w26_17 = {w26_17}; mat4 w26_18 = {w26_18}; mat4 w26_19 = {w26_19}; mat4 w26_20 = {w26_20}; mat4 w26_21 = {w26_21}; mat4 w26_22 = {w26_22}; mat4 w26_23 = {w26_23}; mat4 w26_24 = {w26_24}; vec4 b26 = {b26};
    mat4 w27_1 = {w27_1}; mat4 w27_2 = {w27_2}; mat4 w27_3 = {w27_3}; mat4 w27_4 = {w27_4}; mat4 w27_5 = {w27_5}; mat4 w27_6 = {w27_6}; mat4 w27_7 = {w27_7}; mat4 w27_8 = {w27_8}; mat4 w27_9 = {w27_9}; mat4 w27_10 = {w27_10}; mat4 w27_11 = {w27_11}; mat4 w27_12 = {w27_12}; mat4 w27_13 = {w27_13}; mat4 w27_14 = {w27_14}; mat4 w27_15 = {w27_15}; mat4 w27_16 = {w27_16}; mat4 w27_17 = {w27_17}; mat4 w27_18 = {w27_18}; mat4 w27_19 = {w27_19}; mat4 w27_20 = {w27_20}; mat4 w27_21 = {w27_21}; mat4 w27_22 = {w27_22}; mat4 w27_23 = {w27_23}; mat4 w27_24 = {w27_24}; vec4 b27 = {b27};
    mat4 w28_1 = {w28_1}; mat4 w28_2 = {w28_2}; mat4 w28_3 = {w28_3}; mat4 w28_4 = {w28_4}; mat4 w28_5 = {w28_5}; mat4 w28_6 = {w28_6}; mat4 w28_7 = {w28_7}; mat4 w28_8 = {w28_8}; mat4 w28_9 = {w28_9}; mat4 w28_10 = {w28_10}; mat4 w28_11 = {w28_11}; mat4 w28_12 = {w28_12}; mat4 w28_13 = {w28_13}; mat4 w28_14 = {w28_14}; mat4 w28_15 = {w28_15}; mat4 w28_16 = {w28_16}; mat4 w28_17 = {w28_17}; mat4 w28_18 = {w28_18}; mat4 w28_19 = {w28_19}; mat4 w28_20 = {w28_20}; mat4 w28_21 = {w28_21}; mat4 w28_22 = {w28_22}; mat4 w28_23 = {w28_23}; mat4 w28_24 = {w28_24}; vec4 b28 = {b28};

    // Level 8
    mat4 w29_1 = {w29_1}; mat4 w29_2 = {w29_2}; mat4 w29_3 = {w29_3}; mat4 w29_4 = {w29_4}; mat4 w29_5 = {w29_5}; mat4 w29_6 = {w29_6}; mat4 w29_7 = {w29_7}; mat4 w29_8 = {w29_8}; mat4 w29_9 = {w29_9}; mat4 w29_10 = {w29_10}; mat4 w29_11 = {w29_11}; mat4 w29_12 = {w29_12}; mat4 w29_13 = {w29_13}; mat4 w29_14 = {w29_14}; mat4 w29_15 = {w29_15}; mat4 w29_16 = {w29_16}; mat4 w29_17 = {w29_17}; mat4 w29_18 = {w29_18}; mat4 w29_19 = {w29_19}; mat4 w29_20 = {w29_20}; mat4 w29_21 = {w29_21}; mat4 w29_22 = {w29_22}; mat4 w29_23 = {w29_23}; mat4 w29_24 = {w29_24}; mat4 w29_25 = {w29_25}; mat4 w29_26 = {w29_26}; mat4 w29_27 = {w29_27}; mat4 w29_28 = {w29_28}; vec4 b29 = {b29};
    mat4 w30_1 = {w30_1}; mat4 w30_2 = {w30_2}; mat4 w30_3 = {w30_3}; mat4 w30_4 = {w30_4}; mat4 w30_5 = {w30_5}; mat4 w30_6 = {w30_6}; mat4 w30_7 = {w30_7}; mat4 w30_8 = {w30_8}; mat4 w30_9 = {w30_9}; mat4 w30_10 = {w30_10}; mat4 w30_11 = {w30_11}; mat4 w30_12 = {w30_12}; mat4 w30_13 = {w30_13}; mat4 w30_14 = {w30_14}; mat4 w30_15 = {w30_15}; mat4 w30_16 = {w30_16}; mat4 w30_17 = {w30_17}; mat4 w30_18 = {w30_18}; mat4 w30_19 = {w30_19}; mat4 w30_20 = {w30_20}; mat4 w30_21 = {w30_21}; mat4 w30_22 = {w30_22}; mat4 w30_23 = {w30_23}; mat4 w30_24 = {w30_24}; mat4 w30_25 = {w30_25}; mat4 w30_26 = {w30_26}; mat4 w30_27 = {w30_27}; mat4 w30_28 = {w30_28}; vec4 b30 = {b30};
    mat4 w31_1 = {w31_1}; mat4 w31_2 = {w31_2}; mat4 w31_3 = {w31_3}; mat4 w31_4 = {w31_4}; mat4 w31_5 = {w31_5}; mat4 w31_6 = {w31_6}; mat4 w31_7 = {w31_7}; mat4 w31_8 = {w31_8}; mat4 w31_9 = {w31_9}; mat4 w31_10 = {w31_10}; mat4 w31_11 = {w31_11}; mat4 w31_12 = {w31_12}; mat4 w31_13 = {w31_13}; mat4 w31_14 = {w31_14}; mat4 w31_15 = {w31_15}; mat4 w31_16 = {w31_16}; mat4 w31_17 = {w31_17}; mat4 w31_18 = {w31_18}; mat4 w31_19 = {w31_19}; mat4 w31_20 = {w31_20}; mat4 w31_21 = {w31_21}; mat4 w31_22 = {w31_22}; mat4 w31_23 = {w31_23}; mat4 w31_24 = {w31_24}; mat4 w31_25 = {w31_25}; mat4 w31_26 = {w31_26}; mat4 w31_27 = {w31_27}; mat4 w31_28 = {w31_28}; vec4 b31 = {b31};
    mat4 w32_1 = {w32_1}; mat4 w32_2 = {w32_2}; mat4 w32_3 = {w32_3}; mat4 w32_4 = {w32_4}; mat4 w32_5 = {w32_5}; mat4 w32_6 = {w32_6}; mat4 w32_7 = {w32_7}; mat4 w32_8 = {w32_8}; mat4 w32_9 = {w32_9}; mat4 w32_10 = {w32_10}; mat4 w32_11 = {w32_11}; mat4 w32_12 = {w32_12}; mat4 w32_13 = {w32_13}; mat4 w32_14 = {w32_14}; mat4 w32_15 = {w32_15}; mat4 w32_16 = {w32_16}; mat4 w32_17 = {w32_17}; mat4 w32_18 = {w32_18}; mat4 w32_19 = {w32_19}; mat4 w32_20 = {w32_20}; mat4 w32_21 = {w32_21}; mat4 w32_22 = {w32_22}; mat4 w32_23 = {w32_23}; mat4 w32_24 = {w32_24}; mat4 w32_25 = {w32_25}; mat4 w32_26 = {w32_26}; mat4 w32_27 = {w32_27}; mat4 w32_28 = {w32_28}; vec4 b32 = {b32};

    // Output Layer
    mat4 w33_1 = {w33_1}; mat4 w33_2 = {w33_2}; mat4 w33_3 = {w33_3}; mat4 w33_4 = {w33_4}; mat4 w33_5 = {w33_5}; mat4 w33_6 = {w33_6}; mat4 w33_7 = {w33_7}; mat4 w33_8 = {w33_8}; mat4 w33_9 = {w33_9}; mat4 w33_10 = {w33_10}; mat4 w33_11 = {w33_11}; mat4 w33_12 = {w33_12}; mat4 w33_13 = {w33_13}; mat4 w33_14 = {w33_14}; mat4 w33_15 = {w33_15}; mat4 w33_16 = {w33_16}; mat4 w33_17 = {w33_17}; mat4 w33_18 = {w33_18}; mat4 w33_19 = {w33_19}; mat4 w33_20 = {w33_20}; mat4 w33_21 = {w33_21}; mat4 w33_22 = {w33_22}; mat4 w33_23 = {w33_23}; mat4 w33_24 = {w33_24}; mat4 w33_25 = {w33_25}; mat4 w33_26 = {w33_26}; mat4 w33_27 = {w33_27}; mat4 w33_28 = {w33_28}; mat4 w33_29 = {w33_29}; mat4 w33_30 = {w33_30}; mat4 w33_31 = {w33_31}; mat4 w33_32 = {w33_32}; vec4 b33 = {b33};

    // --- APPLYING NEURAL NETWORK ---
    // Level 1 calculation
    in_buf[0] = mmul(w1_1, in_buf[32]) + mmul(w1_2, in_buf[33]) + b1;
    in_buf[1] = mmul(w2_1, in_buf[32]) + mmul(w2_2, in_buf[33]) + b2;
    in_buf[2] = mmul(w3_1, in_buf[32]) + mmul(w3_2, in_buf[33]) + b3;
    in_buf[3] = mmul(w4_1, in_buf[32]) + mmul(w4_2, in_buf[33]) + b4;
    in_buf[0] = sigmoid(in_buf[0]);     in_buf[1] = sigmoid(in_buf[1]);     in_buf[2] = sigmoid(in_buf[2]);     in_buf[3] = sigmoid(in_buf[3]); 

    // Level 2 calculation
    in_buf[4] = mmul(w5_1, in_buf[0]) + mmul(w5_2, in_buf[1]) + mmul(w5_3, in_buf[2]) + mmul(w5_4, in_buf[3]) + b5;
    in_buf[5] = mmul(w6_1, in_buf[0]) + mmul(w6_2, in_buf[1]) + mmul(w6_3, in_buf[2]) + mmul(w6_4, in_buf[3]) + b6;
    in_buf[6] = mmul(w7_1, in_buf[0]) + mmul(w7_2, in_buf[1]) + mmul(w7_3, in_buf[2]) + mmul(w7_4, in_buf[3]) + b7;
    in_buf[7] = mmul(w8_1, in_buf[0]) + mmul(w8_2, in_buf[1]) + mmul(w8_3, in_buf[2]) + mmul(w8_4, in_buf[3]) + b8;
    in_buf[4] = sigmoid(in_buf[4]);     in_buf[5] = sigmoid(in_buf[5]);     in_buf[6] = sigmoid(in_buf[6]);     in_buf[7] = sigmoid(in_buf[7]); 

    // Level 3 calculation
    in_buf[8] = mmul(w9_1, in_buf[0]) + mmul(w9_2, in_buf[1]) + mmul(w9_3, in_buf[2]) + mmul(w9_4, in_buf[3]) + mmul(w9_5, in_buf[4]) + mmul(w9_6, in_buf[5]) + mmul(w9_7, in_buf[6]) + mmul(w9_8, in_buf[7]) + b9;
    in_buf[9] = mmul(w10_1, in_buf[0]) + mmul(w10_2, in_buf[1]) + mmul(w10_3, in_buf[2]) + mmul(w10_4, in_buf[3]) + mmul(w10_5, in_buf[4]) + mmul(w10_6, in_buf[5]) + mmul(w10_7, in_buf[6]) + mmul(w10_8, in_buf[7]) + b10;
    in_buf[10] = mmul(w11_1, in_buf[0]) + mmul(w11_2, in_buf[1]) + mmul(w11_3, in_buf[2]) + mmul(w11_4, in_buf[3]) + mmul(w11_5, in_buf[4]) + mmul(w11_6, in_buf[5]) + mmul(w11_7, in_buf[6]) + mmul(w11_8, in_buf[7]) + b11;
    in_buf[11] = mmul(w12_1, in_buf[0]) + mmul(w12_2, in_buf[1]) + mmul(w12_3, in_buf[2]) + mmul(w12_4, in_buf[3]) + mmul(w12_5, in_buf[4]) + mmul(w12_6, in_buf[5]) + mmul(w12_7, in_buf[6]) + mmul(w12_8, in_buf[7]) + b12;
    in_buf[8] = sigmoid(in_buf[8]);     in_buf[9] = sigmoid(in_buf[9]);     in_buf[10] = sigmoid(in_buf[10]);     in_buf[11] = sigmoid(in_buf[11]); 

    // Level 4 calculation
    in_buf[12] = mmul(w13_1, in_buf[0]) + mmul(w13_2, in_buf[1]) + mmul(w13_3, in_buf[2]) + mmul(w13_4, in_buf[3]) + mmul(w13_5, in_buf[4]) + mmul(w13_6, in_buf[5]) + mmul(w13_7, in_buf[6]) + mmul(w13_8, in_buf[7]) + mmul(w13_9, in_buf[8]) + mmul(w13_10, in_buf[9]) + mmul(w13_11, in_buf[10]) + mmul(w13_12, in_buf[11]) + b13;
    in_buf[13] = mmul(w14_1, in_buf[0]) + mmul(w14_2, in_buf[1]) + mmul(w14_3, in_buf[2]) + mmul(w14_4, in_buf[3]) + mmul(w14_5, in_buf[4]) + mmul(w14_6, in_buf[5]) + mmul(w14_7, in_buf[6]) + mmul(w14_8, in_buf[7]) + mmul(w14_9, in_buf[8]) + mmul(w14_10, in_buf[9]) + mmul(w14_11, in_buf[10]) + mmul(w14_12, in_buf[11]) + b14;
    in_buf[14] = mmul(w15_1, in_buf[0]) + mmul(w15_2, in_buf[1]) + mmul(w15_3, in_buf[2]) + mmul(w15_4, in_buf[3]) + mmul(w15_5, in_buf[4]) + mmul(w15_6, in_buf[5]) + mmul(w15_7, in_buf[6]) + mmul(w15_8, in_buf[7]) + mmul(w15_9, in_buf[8]) + mmul(w15_10, in_buf[9]) + mmul(w15_11, in_buf[10]) + mmul(w15_12, in_buf[11]) + b15;
    in_buf[15] = mmul(w16_1, in_buf[0]) + mmul(w16_2, in_buf[1]) + mmul(w16_3, in_buf[2]) + mmul(w16_4, in_buf[3]) + mmul(w16_5, in_buf[4]) + mmul(w16_6, in_buf[5]) + mmul(w16_7, in_buf[6]) + mmul(w16_8, in_buf[7]) + mmul(w16_9, in_buf[8]) + mmul(w16_10, in_buf[9]) + mmul(w16_11, in_buf[10]) + mmul(w16_12, in_buf[11]) + b16;
    in_buf[12] = sigmoid(in_buf[12]);     in_buf[13] = sigmoid(in_buf[13]);     in_buf[14] = sigmoid(in_buf[14]);     in_buf[15] = sigmoid(in_buf[15]); 

    // Level 5 calculation
    in_buf[16] = mmul(w17_1, in_buf[0]) + mmul(w17_2, in_buf[1]) + mmul(w17_3, in_buf[2]) + mmul(w17_4, in_buf[3]) + mmul(w17_5, in_buf[4]) + mmul(w17_6, in_buf[5]) + mmul(w17_7, in_buf[6]) + mmul(w17_8, in_buf[7]) + mmul(w17_9, in_buf[8]) + mmul(w17_10, in_buf[9]) + mmul(w17_11, in_buf[10]) + mmul(w17_12, in_buf[11]) + mmul(w17_13, in_buf[12]) + mmul(w17_14, in_buf[13]) + mmul(w17_15, in_buf[14]) + mmul(w17_16, in_buf[15]) + b17;
    in_buf[17] = mmul(w18_1, in_buf[0]) + mmul(w18_2, in_buf[1]) + mmul(w18_3, in_buf[2]) + mmul(w18_4, in_buf[3]) + mmul(w18_5, in_buf[4]) + mmul(w18_6, in_buf[5]) + mmul(w18_7, in_buf[6]) + mmul(w18_8, in_buf[7]) + mmul(w18_9, in_buf[8]) + mmul(w18_10, in_buf[9]) + mmul(w18_11, in_buf[10]) + mmul(w18_12, in_buf[11]) + mmul(w18_13, in_buf[12]) + mmul(w18_14, in_buf[13]) + mmul(w18_15, in_buf[14]) + mmul(w18_16, in_buf[15]) + b18;
    in_buf[18] = mmul(w19_1, in_buf[0]) + mmul(w19_2, in_buf[1]) + mmul(w19_3, in_buf[2]) + mmul(w19_4, in_buf[3]) + mmul(w19_5, in_buf[4]) + mmul(w19_6, in_buf[5]) + mmul(w19_7, in_buf[6]) + mmul(w19_8, in_buf[7]) + mmul(w19_9, in_buf[8]) + mmul(w19_10, in_buf[9]) + mmul(w19_11, in_buf[10]) + mmul(w19_12, in_buf[11]) + mmul(w19_13, in_buf[12]) + mmul(w19_14, in_buf[13]) + mmul(w19_15, in_buf[14]) + mmul(w19_16, in_buf[15]) + b19;
    in_buf[19] = mmul(w20_1, in_buf[0]) + mmul(w20_2, in_buf[1]) + mmul(w20_3, in_buf[2]) + mmul(w20_4, in_buf[3]) + mmul(w20_5, in_buf[4]) + mmul(w20_6, in_buf[5]) + mmul(w20_7, in_buf[6]) + mmul(w20_8, in_buf[7]) + mmul(w20_9, in_buf[8]) + mmul(w20_10, in_buf[9]) + mmul(w20_11, in_buf[10]) + mmul(w20_12, in_buf[11]) + mmul(w20_13, in_buf[12]) + mmul(w20_14, in_buf[13]) + mmul(w20_15, in_buf[14]) + mmul(w20_16, in_buf[15]) + b20;
    in_buf[16] = sigmoid(in_buf[16]);     in_buf[17] = sigmoid(in_buf[17]);     in_buf[18] = sigmoid(in_buf[18]);     in_buf[19] = sigmoid(in_buf[19]); 

    // Level 6 calculation
    in_buf[20] = mmul(w21_1, in_buf[0]) + mmul(w21_2, in_buf[1]) + mmul(w21_3, in_buf[2]) + mmul(w21_4, in_buf[3]) + mmul(w21_5, in_buf[4]) + mmul(w21_6, in_buf[5]) + mmul(w21_7, in_buf[6]) + mmul(w21_8, in_buf[7]) + mmul(w21_9, in_buf[8]) + mmul(w21_10, in_buf[9]) + mmul(w21_11, in_buf[10]) + mmul(w21_12, in_buf[11]) + mmul(w21_13, in_buf[12]) + mmul(w21_14, in_buf[13]) + mmul(w21_15, in_buf[14]) + mmul(w21_16, in_buf[15]) + mmul(w21_17, in_buf[16]) + mmul(w21_18, in_buf[17]) + mmul(w21_19, in_buf[18]) + mmul(w21_20, in_buf[19]) + b21;
    in_buf[21] = mmul(w22_1, in_buf[0]) + mmul(w22_2, in_buf[1]) + mmul(w22_3, in_buf[2]) + mmul(w22_4, in_buf[3]) + mmul(w22_5, in_buf[4]) + mmul(w22_6, in_buf[5]) + mmul(w22_7, in_buf[6]) + mmul(w22_8, in_buf[7]) + mmul(w22_9, in_buf[8]) + mmul(w22_10, in_buf[9]) + mmul(w22_11, in_buf[10]) + mmul(w22_12, in_buf[11]) + mmul(w22_13, in_buf[12]) + mmul(w22_14, in_buf[13]) + mmul(w22_15, in_buf[14]) + mmul(w22_16, in_buf[15]) + mmul(w22_17, in_buf[16]) + mmul(w22_18, in_buf[17]) + mmul(w22_19, in_buf[18]) + mmul(w22_20, in_buf[19]) + b22;
    in_buf[22] = mmul(w23_1, in_buf[0]) + mmul(w23_2, in_buf[1]) + mmul(w23_3, in_buf[2]) + mmul(w23_4, in_buf[3]) + mmul(w23_5, in_buf[4]) + mmul(w23_6, in_buf[5]) + mmul(w23_7, in_buf[6]) + mmul(w23_8, in_buf[7]) + mmul(w23_9, in_buf[8]) + mmul(w23_10, in_buf[9]) + mmul(w23_11, in_buf[10]) + mmul(w23_12, in_buf[11]) + mmul(w23_13, in_buf[12]) + mmul(w23_14, in_buf[13]) + mmul(w23_15, in_buf[14]) + mmul(w23_16, in_buf[15]) + mmul(w23_17, in_buf[16]) + mmul(w23_18, in_buf[17]) + mmul(w23_19, in_buf[18]) + mmul(w23_20, in_buf[19]) + b23;
    in_buf[23] = mmul(w24_1, in_buf[0]) + mmul(w24_2, in_buf[1]) + mmul(w24_3, in_buf[2]) + mmul(w24_4, in_buf[3]) + mmul(w24_5, in_buf[4]) + mmul(w24_6, in_buf[5]) + mmul(w24_7, in_buf[6]) + mmul(w24_8, in_buf[7]) + mmul(w24_9, in_buf[8]) + mmul(w24_10, in_buf[9]) + mmul(w24_11, in_buf[10]) + mmul(w24_12, in_buf[11]) + mmul(w24_13, in_buf[12]) + mmul(w24_14, in_buf[13]) + mmul(w24_15, in_buf[14]) + mmul(w24_16, in_buf[15]) + mmul(w24_17, in_buf[16]) + mmul(w24_18, in_buf[17]) + mmul(w24_19, in_buf[18]) + mmul(w24_20, in_buf[19]) + b24;
    in_buf[20] = sigmoid(in_buf[20]);     in_buf[21] = sigmoid(in_buf[21]);     in_buf[22] = sigmoid(in_buf[22]);     in_buf[23] = sigmoid(in_buf[23]); 

    // Level 7 calculation
    in_buf[24] = mmul(w25_1, in_buf[0]) + mmul(w25_2, in_buf[1]) + mmul(w25_3, in_buf[2]) + mmul(w25_4, in_buf[3]) + mmul(w25_5, in_buf[4]) + mmul(w25_6, in_buf[5]) + mmul(w25_7, in_buf[6]) + mmul(w25_8, in_buf[7]) + mmul(w25_9, in_buf[8]) + mmul(w25_10, in_buf[9]) + mmul(w25_11, in_buf[10]) + mmul(w25_12, in_buf[11]) + mmul(w25_13, in_buf[12]) + mmul(w25_14, in_buf[13]) + mmul(w25_15, in_buf[14]) + mmul(w25_16, in_buf[15]) + mmul(w25_17, in_buf[16]) + mmul(w25_18, in_buf[17]) + mmul(w25_19, in_buf[18]) + mmul(w25_20, in_buf[19]) + mmul(w25_21, in_buf[20]) + mmul(w25_22, in_buf[21]) + mmul(w25_23, in_buf[22]) + mmul(w25_24, in_buf[23]) + b25;
    in_buf[25] = mmul(w26_1, in_buf[0]) + mmul(w26_2, in_buf[1]) + mmul(w26_3, in_buf[2]) + mmul(w26_4, in_buf[3]) + mmul(w26_5, in_buf[4]) + mmul(w26_6, in_buf[5]) + mmul(w26_7, in_buf[6]) + mmul(w26_8, in_buf[7]) + mmul(w26_9, in_buf[8]) + mmul(w26_10, in_buf[9]) + mmul(w26_11, in_buf[10]) + mmul(w26_12, in_buf[11]) + mmul(w26_13, in_buf[12]) + mmul(w26_14, in_buf[13]) + mmul(w26_15, in_buf[14]) + mmul(w26_16, in_buf[15]) + mmul(w26_17, in_buf[16]) + mmul(w26_18, in_buf[17]) + mmul(w26_19, in_buf[18]) + mmul(w26_20, in_buf[19]) + mmul(w26_21, in_buf[20]) + mmul(w26_22, in_buf[21]) + mmul(w26_23, in_buf[22]) + mmul(w26_24, in_buf[23]) + b26;
    in_buf[26] = mmul(w27_1, in_buf[0]) + mmul(w27_2, in_buf[1]) + mmul(w27_3, in_buf[2]) + mmul(w27_4, in_buf[3]) + mmul(w27_5, in_buf[4]) + mmul(w27_6, in_buf[5]) + mmul(w27_7, in_buf[6]) + mmul(w27_8, in_buf[7]) + mmul(w27_9, in_buf[8]) + mmul(w27_10, in_buf[9]) + mmul(w27_11, in_buf[10]) + mmul(w27_12, in_buf[11]) + mmul(w27_13, in_buf[12]) + mmul(w27_14, in_buf[13]) + mmul(w27_15, in_buf[14]) + mmul(w27_16, in_buf[15]) + mmul(w27_17, in_buf[16]) + mmul(w27_18, in_buf[17]) + mmul(w27_19, in_buf[18]) + mmul(w27_20, in_buf[19]) + mmul(w27_21, in_buf[20]) + mmul(w27_22, in_buf[21]) + mmul(w27_23, in_buf[22]) + mmul(w27_24, in_buf[23]) + b27;
    in_buf[27] = mmul(w28_1, in_buf[0]) + mmul(w28_2, in_buf[1]) + mmul(w28_3, in_buf[2]) + mmul(w28_4, in_buf[3]) + mmul(w28_5, in_buf[4]) + mmul(w28_6, in_buf[5]) + mmul(w28_7, in_buf[6]) + mmul(w28_8, in_buf[7]) + mmul(w28_9, in_buf[8]) + mmul(w28_10, in_buf[9]) + mmul(w28_11, in_buf[10]) + mmul(w28_12, in_buf[11]) + mmul(w28_13, in_buf[12]) + mmul(w28_14, in_buf[13]) + mmul(w28_15, in_buf[14]) + mmul(w28_16, in_buf[15]) + mmul(w28_17, in_buf[16]) + mmul(w28_18, in_buf[17]) + mmul(w28_19, in_buf[18]) + mmul(w28_20, in_buf[19]) + mmul(w28_21, in_buf[20]) + mmul(w28_22, in_buf[21]) + mmul(w28_23, in_buf[22]) + mmul(w28_24, in_buf[23]) + b28;
    in_buf[24] = sigmoid(in_buf[24]);     in_buf[25] = sigmoid(in_buf[25]);     in_buf[26] = sigmoid(in_buf[26]);     in_buf[27] = sigmoid(in_buf[27]); 

    // Level 8 calculation
    in_buf[28] = mmul(w29_1, in_buf[0]) + mmul(w29_2, in_buf[1]) + mmul(w29_3, in_buf[2]) + mmul(w29_4, in_buf[3]) + mmul(w29_5, in_buf[4]) + mmul(w29_6, in_buf[5]) + mmul(w29_7, in_buf[6]) + mmul(w29_8, in_buf[7]) + mmul(w29_9, in_buf[8]) + mmul(w29_10, in_buf[9]) + mmul(w29_11, in_buf[10]) + mmul(w29_12, in_buf[11]) + mmul(w29_13, in_buf[12]) + mmul(w29_14, in_buf[13]) + mmul(w29_15, in_buf[14]) + mmul(w29_16, in_buf[15]) + mmul(w29_17, in_buf[16]) + mmul(w29_18, in_buf[17]) + mmul(w29_19, in_buf[18]) + mmul(w29_20, in_buf[19]) + mmul(w29_21, in_buf[20]) + mmul(w29_22, in_buf[21]) + mmul(w29_23, in_buf[22]) + mmul(w29_24, in_buf[23]) + mmul(w29_25, in_buf[24]) + mmul(w29_26, in_buf[25]) + mmul(w29_27, in_buf[26]) + mmul(w29_28, in_buf[27]) + b29;
    in_buf[29] = mmul(w30_1, in_buf[0]) + mmul(w30_2, in_buf[1]) + mmul(w30_3, in_buf[2]) + mmul(w30_4, in_buf[3]) + mmul(w30_5, in_buf[4]) + mmul(w30_6, in_buf[5]) + mmul(w30_7, in_buf[6]) + mmul(w30_8, in_buf[7]) + mmul(w30_9, in_buf[8]) + mmul(w30_10, in_buf[9]) + mmul(w30_11, in_buf[10]) + mmul(w30_12, in_buf[11]) + mmul(w30_13, in_buf[12]) + mmul(w30_14, in_buf[13]) + mmul(w30_15, in_buf[14]) + mmul(w30_16, in_buf[15]) + mmul(w30_17, in_buf[16]) + mmul(w30_18, in_buf[17]) + mmul(w30_19, in_buf[18]) + mmul(w30_20, in_buf[19]) + mmul(w30_21, in_buf[20]) + mmul(w30_22, in_buf[21]) + mmul(w30_23, in_buf[22]) + mmul(w30_24, in_buf[23]) + mmul(w30_25, in_buf[24]) + mmul(w30_26, in_buf[25]) + mmul(w30_27, in_buf[26]) + mmul(w30_28, in_buf[27]) + b30;
    in_buf[30] = mmul(w31_1, in_buf[0]) + mmul(w31_2, in_buf[1]) + mmul(w31_3, in_buf[2]) + mmul(w31_4, in_buf[3]) + mmul(w31_5, in_buf[4]) + mmul(w31_6, in_buf[5]) + mmul(w31_7, in_buf[6]) + mmul(w31_8, in_buf[7]) + mmul(w31_9, in_buf[8]) + mmul(w31_10, in_buf[9]) + mmul(w31_11, in_buf[10]) + mmul(w31_12, in_buf[11]) + mmul(w31_13, in_buf[12]) + mmul(w31_14, in_buf[13]) + mmul(w31_15, in_buf[14]) + mmul(w31_16, in_buf[15]) + mmul(w31_17, in_buf[16]) + mmul(w31_18, in_buf[17]) + mmul(w31_19, in_buf[18]) + mmul(w31_20, in_buf[19]) + mmul(w31_21, in_buf[20]) + mmul(w31_22, in_buf[21]) + mmul(w31_23, in_buf[22]) + mmul(w31_24, in_buf[23]) + mmul(w31_25, in_buf[24]) + mmul(w31_26, in_buf[25]) + mmul(w31_27, in_buf[26]) + mmul(w31_28, in_buf[27]) + b31;
    in_buf[31] = mmul(w32_1, in_buf[0]) + mmul(w32_2, in_buf[1]) + mmul(w32_3, in_buf[2]) + mmul(w32_4, in_buf[3]) + mmul(w32_5, in_buf[4]) + mmul(w32_6, in_buf[5]) + mmul(w32_7, in_buf[6]) + mmul(w32_8, in_buf[7]) + mmul(w32_9, in_buf[8]) + mmul(w32_10, in_buf[9]) + mmul(w32_11, in_buf[10]) + mmul(w32_12, in_buf[11]) + mmul(w32_13, in_buf[12]) + mmul(w32_14, in_buf[13]) + mmul(w32_15, in_buf[14]) + mmul(w32_16, in_buf[15]) + mmul(w32_17, in_buf[16]) + mmul(w32_18, in_buf[17]) + mmul(w32_19, in_buf[18]) + mmul(w32_20, in_buf[19]) + mmul(w32_21, in_buf[20]) + mmul(w32_22, in_buf[21]) + mmul(w32_23, in_buf[22]) + mmul(w32_24, in_buf[23]) + mmul(w32_25, in_buf[24]) + mmul(w32_26, in_buf[25]) + mmul(w32_27, in_buf[26]) + mmul(w32_28, in_buf[27]) + b32;
    in_buf[28] = sigmoid(in_buf[28]);     in_buf[29] = sigmoid(in_buf[29]);     in_buf[30] = sigmoid(in_buf[30]);     in_buf[31] = sigmoid(in_buf[31]); 

    // Output Layer calculation
    in_buf[0] = mmul(w33_1, in_buf[0]) + mmul(w33_2, in_buf[1]) + mmul(w33_3, in_buf[2]) + mmul(w33_4, in_buf[3]) + mmul(w33_5, in_buf[4]) + mmul(w33_6, in_buf[5]) + mmul(w33_7, in_buf[6]) + mmul(w33_8, in_buf[7]) + mmul(w33_9, in_buf[8]) + mmul(w33_10, in_buf[9]) + mmul(w33_11, in_buf[10]) + mmul(w33_12, in_buf[11]) + mmul(w33_13, in_buf[12]) + mmul(w33_14, in_buf[13]) + mmul(w33_15, in_buf[14]) + mmul(w33_16, in_buf[15]) + mmul(w33_17, in_buf[16]) + mmul(w33_18, in_buf[17]) + mmul(w33_19, in_buf[18]) + mmul(w33_20, in_buf[19]) + mmul(w33_21, in_buf[20]) + mmul(w33_22, in_buf[21]) + mmul(w33_23, in_buf[22]) + mmul(w33_24, in_buf[23]) + mmul(w33_25, in_buf[24]) + mmul(w33_26, in_buf[25]) + mmul(w33_27, in_buf[26]) + mmul(w33_28, in_buf[27]) + mmul(w33_29, in_buf[28]) + mmul(w33_30, in_buf[29]) + mmul(w33_31, in_buf[30]) + mmul(w33_32, in_buf[31]) + b33;
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
        default=128,
        help="Training resolution (larger = more GPU work, default: 256)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.005, help="Learning rate (default: 0.002)"
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

    # Generate persistent random colored "Perlin-like" noise (low resolution scaled up to create blobs)
    noise_res = max(4, args.size // 16)
    np_noise = np.random.rand(noise_res, noise_res, 4).astype(np.float32)
    
    # We use cv2.resize to smoothly interpolate the low-res random matrix up to full size
    # This creates a very organic, wavy, colorful "Perlin" noise look.
    perlin_like_noise = cv2.resize(np_noise, (args.size, args.size), interpolation=cv2.INTER_CUBIC)
    # Ensure full alpha (opacity) for the noise
    perlin_like_noise[..., 3] = 1.0 
    
    noise_target_flat = torch.tensor(perlin_like_noise).view(-1, 4).to(device)
    pretrain_steps = 300 # Train purely on organic random noise for first N steps

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

            # --- Pre-train Phase Logic ---
            # If we are under 'pretrain_steps', we use the Perlin-like noise as the target.
            # Otherwise, we switch to the actual image 'targets_flat'.
            current_target = noise_target_flat if step < pretrain_steps else targets_flat

            mse_loss = criterion(pred_flat[:, :4], current_target)
            if edge_criterion is not None:
                edge_loss = edge_criterion(pred_flat[:, :4], current_target, args.size)
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

    export_weights(model, "trained_cppn_32.glsl", seeds=seeds)


if __name__ == "__main__":
    main()

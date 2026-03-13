import pygame
from OpenGL.GL import (
    glGetUniformLocation,
    glActiveTexture,
    glBindTexture,
    glClear,
    glUseProgram,
    glUniform1f,
    glUniform3fv,
    glUniform4f,
    glUniform1i,
    glBegin,
    glTexCoord2f,
    glVertex3f,
    glEnd,
    glClearColor,
    glGenTextures,
    glTexImage2D,
    glTexParameteri,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_TRIANGLES,
    GL_VERTEX_SHADER,
    GL_FRAGMENT_SHADER,
    GL_RGB,
    GL_UNSIGNED_BYTE,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_MAG_FILTER,
    GL_LINEAR,
)
from OpenGL.GL.shaders import compileProgram, compileShader
import time
import numpy as np
import tkinter as tk
from tkinter import filedialog

vertex_shader = """
#version 120

varying vec2 uv;
varying vec2 v_TexCoord;
varying vec3 posBuff;
void main(){
    gl_Position = gl_Vertex;
    posBuff = gl_Vertex.xyz;
    uv = (gl_Vertex.xy * 0.5) + 0.5;
    // uv = gl_Vertex.xy;
    v_TexCoord = uv;
}
"""


def read_shader_file(file_path):
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading shader file: {e}")
        return None


import os

root = tk.Tk()
root.withdraw()

# Ensure we start looking in the current directory
initial_dir = os.path.abspath(".")
file_path = tk.filedialog.askopenfilename(initialdir=initial_dir, filetypes=[("GLSL files", "*.glsl *.frag *.fs")])  # type: ignore

fragment_shader = read_shader_file(file_path)

import json
import re


def get_uniform_defaults(shader_code):
    defaults = {}
    if not shader_code:
        return defaults
    # Match uniform float <name>; // {json} allowing for line breaks
    pattern = r"uniform\s+float\s+([a-zA-Z0-9_]+)\s*;\s*//\s*(\{.*?\})"
    for match in re.finditer(pattern, shader_code):
        var_name = match.group(1)
        json_str = match.group(2)
        try:
            data = json.loads(json_str)
            if "default" in data:
                defaults[var_name] = float(data["default"])
        except Exception as e:
            print(f"Failed to parse default for {var_name}: {e}")
    return defaults


uniform_defaults = get_uniform_defaults(fragment_shader)

resolution = (960, 540)
resolution_array = np.array([resolution[0], resolution[1], 1.0], dtype=np.float32)
start_time = time.time()


def display():
    global texture
    # Get Uniform location
    g_Time = glGetUniformLocation(shader_program, "g_Time")
    g_Screen = glGetUniformLocation(shader_program, "g_Screen")
    iMouse = glGetUniformLocation(shader_program, "iMouse")
    iChannel0 = glGetUniformLocation(shader_program, "iChannel0")

    # capture mouse position
    mouse_pos = pygame.mouse.get_pos()
    mouse_pos = (mouse_pos[0], resolution[1] - mouse_pos[1])

    # Bind the existing texture instead of loading it every frame (avoids memory leak)
    if texture is not None:
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture)

    # draw triangles to fill screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # type: ignore
    glUseProgram(shader_program)

    glUniform1f(g_Time, time.time() - start_time)
    glUniform3fv(g_Screen, 1, resolution_array)
    glUniform4f(iMouse, mouse_pos[0], mouse_pos[1], 0, 0)
    glUniform1i(iChannel0, 0)

    # Check if texture1 exists defined in previous context and apply it too, wait, the old file was using texture1 occasionally but here we are in display. Let's just apply the parsed uniforms:
    if "iChannel1" in globals():
        pass  # just keeping the structure intact

    for name, value in uniform_defaults.items():
        loc = glGetUniformLocation(shader_program, name)
        if loc != -1:
            glUniform1f(loc, value)

    glBegin(GL_TRIANGLES)

    glTexCoord2f(0.0, 0.0)
    glVertex3f(-1.0, -1.0, 0)

    glTexCoord2f(1.0, 0.0)
    glVertex3f(1.0, -1.0, 0)

    glTexCoord2f(1.0, 1.0)
    glVertex3f(1.0, 1.0, 0)

    glTexCoord2f(0.0, 0.0)
    glVertex3f(-1.0, -1.0, 0)

    glTexCoord2f(1.0, 1.0)
    glVertex3f(1.0, 1.0, 0)

    glTexCoord2f(0.0, 1.0)
    glVertex3f(-1.0, 1.0, 0)

    glEnd()

    # Unbind texture and flip to update screen
    glUseProgram(0)
    pygame.display.flip()


def init():
    global texture
    glClearColor(0.0, 0.0, 0.0, 1.0)

    # Load image once and generate texture here to prevent RAM/VRAM leaks
    try:
        image = pygame.image.load("./shaders/ComfyUI_00019_.png")
        image_data = pygame.image.tostring(image, "RGB")

        texture = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,
            image.get_width(),
            image.get_height(),
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            image_data,
        )

        # Set texture parameters so it renders correctly
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    except Exception as e:
        print(f"Error loading texture in init: {e}")
        texture = None


def main():
    global shader_program

    pygame.init()
    pygame.display.set_mode(resolution, pygame.DOUBLEBUF | pygame.OPENGL)

    # Compile shaders
    shader_program = compileProgram(
        compileShader(vertex_shader, GL_VERTEX_SHADER),
        compileShader(fragment_shader, GL_FRAGMENT_SHADER),
    )

    g_Time = glGetUniformLocation(shader_program, "g_Time")

    # Initialize OpenGL
    init()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        display()


if __name__ == "__main__":
    main()

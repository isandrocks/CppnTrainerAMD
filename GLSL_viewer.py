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
import json
import re

# Initialize Tkinter companion window
root = tk.Tk()
root.title("GLSL Viewer Controls")
root.geometry("300x120")

# Instead of withdrawing completely, we keep it visible for the menu.
initial_dir = os.path.abspath(".")
file_path = tk.filedialog.askopenfilename(initialdir=initial_dir, filetypes=[("GLSL files", "*.glsl *.frag *.fs")])  # type: ignore
if not file_path:
    print("No file selected. Exiting.")
    exit(0)

fragment_shader = read_shader_file(file_path)
shader_program = None
uniform_defaults = {}
texture = None
running = True

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

def reload_file(*args):
    global fragment_shader, shader_program, uniform_defaults
    if not file_path:
        return
    new_shader = read_shader_file(file_path)
    if new_shader:
        try:
            new_program = compileProgram(
                compileShader(vertex_shader, GL_VERTEX_SHADER),
                compileShader(new_shader, GL_FRAGMENT_SHADER),
            )
            shader_program = new_program
            fragment_shader = new_shader
            uniform_defaults = get_uniform_defaults(fragment_shader)
            print(f"Reloaded {file_path}")
        except Exception as e:
            print(f"Error compiling shader: {e}")

def open_file():
    global file_path
    new_file = filedialog.askopenfilename(initialdir=initial_dir, filetypes=[("GLSL files", "*.glsl *.frag *.fs")])
    if new_file:
        file_path = new_file
        reload_file()

def quit_app():
    global running
    running = False

# Setup Menu Bar
menu_bar = tk.Menu(root)
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Open", command=open_file)
file_menu.add_command(label="Reload File (F5)", command=reload_file)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=quit_app)
menu_bar.add_cascade(label="File", menu=file_menu)
root.config(menu=menu_bar)

# Keep the control panel on top
root.attributes('-topmost', 1)

tk.Label(root, text="GLSL Viewer Controls\nUse File menu or press F5 in viewer to reload.", pady=20).pack()

root.protocol("WM_DELETE_WINDOW", quit_app)

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
    global shader_program, running

    import os
    # Set Pygame window to always on top before initialization
    os.environ['SDL_VIDEO_WINDOW_POS'] = '100,100'

    pygame.init()
    
    # Enable always-on-top mode for Windows (requires SDL2)
    import platform
    if platform.system() == 'Windows':
        import ctypes
        from ctypes import wintypes
    
    pygame.display.set_mode(resolution, pygame.DOUBLEBUF | pygame.OPENGL)

    # Compile shaders
    shader_program = compileProgram(
        compileShader(vertex_shader, GL_VERTEX_SHADER),
        compileShader(fragment_shader, GL_FRAGMENT_SHADER),
    )

    g_Time = glGetUniformLocation(shader_program, "g_Time")

    # Initialize OpenGL
    init()
    
    # Apply Windows specific always-on-top using ctypes
    # Doing it after init() to ensure OpenGL context changes don't reset it
    import platform
    if platform.system() == 'Windows':
        import ctypes
        from ctypes import wintypes
        HWND_TOPMOST = -1
        SWP_NOSIZE = 0x0001
        SWP_NOMOVE = 0x0002
        hwnd = pygame.display.get_wm_info()["window"]
        
        SetWindowPos = ctypes.windll.user32.SetWindowPos
        SetWindowPos.restype = wintypes.BOOL
        SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
        SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)

    while running:
        # Pump the Tkinter event loop
        try:
            root.update()
        except tk.TclError:
            running = False # Tk window closed

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F5:
                    reload_file()
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if running:
            display()
            
    pygame.quit()
    try:
        root.destroy()
    except tk.TclError:
        pass


if __name__ == "__main__":
    main()

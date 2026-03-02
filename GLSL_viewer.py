import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import time
import numpy as np
import tkinter as tk
from tkinter import filedialog

vertex_shader = """
#version 300 es

layout(location = 0) in vec4 in_position;
layout(location = 1) in vec2 in_uv;

out vec2 uv;

void main(){
    gl_Position = in_position;

    uv = in_uv;
}
"""

def read_shader_file(file_path):
  try:
    with open(file_path, 'r') as file:
      return file.read()
  except Exception as e:
    print(f"Error reading shader file: {e}")
    return None

import os

root = tk.Tk()
root.withdraw()

# Ensure we start looking in the current directory
initial_dir = os.path.abspath(".")
file_path = tk.filedialog.askopenfilename(initialdir=initial_dir, filetypes=[("GLSL files", "*.glsl *.frag *.fs")])

fragment_shader = read_shader_file(file_path)

resolution = (800, 600)
resolution_array = np.array(resolution, dtype=np.float32)

def display():
  global texture
  # Get Uniform location
  iTime = glGetUniformLocation(shader_program, "iTime")
  iResolution = glGetUniformLocation(shader_program, "iResolution")
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
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
  glUseProgram(shader_program)
  glBegin(GL_TRIANGLES)
  glVertex3f(-1.0, -1.0, 0)
  glVertex3f(1.0, -1.0, 0)
  glVertex3f(1.0, 1.0, 0)
  
  glVertex3f(-1.0, -1.0, 0)
  glVertex3f(1.0, 1.0, 0)
  glVertex3f(-1.0, 1.0, 0)
  glEnd()

  # Update uniform
  glUniform1f(iTime, time.thread_time())
  glUniform2fv(iResolution, 1, resolution_array)
  glUniform4f(iMouse, mouse_pos[0], mouse_pos[1], 0, 0)
  glUniform1i(iChannel0, 0)

  # Unbind texture and flip to update screen
  glUseProgram(0)
  pygame.display.flip()


def init():
  global texture
  glClearColor(0.0, 0.0, 0.0, 1.0)
  
  # Load image once and generate texture here to prevent RAM/VRAM leaks
  try:
      image = pygame.image.load('./shaders/ComfyUI_00019_.png')
      image_data = pygame.image.tostring(image, "RGB")

      texture = glGenTextures(1)
      glActiveTexture(GL_TEXTURE0)
      glBindTexture(GL_TEXTURE_2D, texture)
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.get_width(), image.get_height(), 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)
      
      # Set texture parameters so it renders correctly
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
  except Exception as e:
      print(f"Error loading texture in init: {e}")
      texture = None

def main():
  global shader_program
  
  pygame.init()
  pygame.display.set_mode(resolution, DOUBLEBUF | OPENGL)

  # Compile shaders
  shader_program = compileProgram(
    compileShader(vertex_shader, GL_VERTEX_SHADER),
    compileShader(fragment_shader, GL_FRAGMENT_SHADER)
  )

  iTime = glGetUniformLocation(shader_program, "iTime")

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

#!/usr/bin/env python3

def matrix_shape(matrix):
  shape = []
  
  if value := matrix:
    shape.append(len(value))
    while isinstance(value[0], list):
      shape.append(len(value[0]))
      value = value[0]
  
  return shape
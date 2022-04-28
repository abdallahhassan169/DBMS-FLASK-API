# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 18:10:35 2021

@author: abdal
"""
import cv2
photo=cv2.imread('animal.jpg.jpg')
gray=cv2.cvtColor(photo,cv2.COLOR_BGR2GRAY)
cv2.imshow('original',photo)
cv2.imshow('gray',gray)
cv2.waitKey()
cv2.destroyAllWindows()
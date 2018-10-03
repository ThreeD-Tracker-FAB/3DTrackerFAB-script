"""
[Python scripts for 3DTracker-FAB (www.3dtracker.org)]
Example 02: Converting 3D position to 2D

This is a script demonstrating how to convert a 3D point to a 2D position in a RGB image.
The type of conversion is useful for estimating keypoint positions (e.g., body parts center)
in the 2D image and following detailed image analysis around the point.

This example plots the 3D coordinate origin and Z axis on the RGB image

Date last modified: 2018.10.03
"""

import numpy as np
import cv2

import contextlib

import pyqtgraph as pg
import pyqtgraph.opengl as gl

import lib3dtracker as tdt  # 3DTracker-FAB python library

fname_metadata = './example data/dual_d435_01/dual_d435_01.metadata.xml' # metadata file path

with contextlib.closing(tdt.DataReader(fname_metadata)) as d:   # open data using 'with statement'

    i_frame = 10;   # video frame number to process

    # 3D point to convert
    point_3d_origin = [0, 0, 0] 
    point_3d_z = [0, 0, -0.2] 

    for i_cam in range(2):

        # read RGB image
        [frame_rgb, frame_d] = d.get_rgbd_frame(i_frame, i_cam)

        # convert 3D points to 2D
        point_2d_origin = d.project_point_to_rgb(point_3d_origin, i_cam)
        point_2d_z = d.project_point_to_rgb(point_3d_z, i_cam)

        # show the 2D positions (origin and z axis)
        cv2.ellipse(frame_rgb, tuple(point_2d_origin), (10, 10), 0, 0, 360, (0, 0, 255), 2)
        cv2.line(frame_rgb, tuple(point_2d_origin), tuple(point_2d_z), (0, 255, 255), 2)
        cv2.imshow('rgb '+ str(i_cam+1), frame_rgb)

    # 3D plot to confirm -----------------------------------

    # prepare for plotting
    app=pg.QtGui.QApplication([])   
    w = gl.GLViewWidget()   
    
    # read and plot merged point cloud
    pc = d.get_mrgpc_frame(i_frame) 
    tdt.plot_pc(pc, w, 4)  

    # plot axis
    g=gl.GLAxisItem()
    w.addItem(g)

    # show the plot
    w.setCameraPosition(distance = 0.5)
    w.show()
    print('Close the window to quit.')
    pg.QtGui.QApplication.exec_()  



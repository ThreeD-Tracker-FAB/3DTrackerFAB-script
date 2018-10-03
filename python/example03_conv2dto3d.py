"""
[Python scripts for 3DTracker-FAB (www.3dtracker.org)]
Example 03: Converting 2D position to 3D

This is a script demonstrating how to convert 2D positions in a ROI in a RGB image to 3D.
The type of conversion is useful for using 2D image based object detection/tracking 
algorithms to obtain the corresponding 3D object position/trace.

The example plot 3D points in the ROIs surrouding a can in the 2D images

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

    # show camera 1 RGB image and ROI
    roi_cam1 = [120, 70, 40, 80] # ROI; left, top, width, height
    [frame_rgb, frame_d] = d.get_rgbd_frame(i_frame, 0)
    cv2.rectangle(frame_rgb, tuple(roi_cam1[0:2]), (roi_cam1[0]+roi_cam1[2], roi_cam1[1]+roi_cam1[3]), (0, 0, 255), 2)
    cv2.imshow('rgb1', frame_rgb)

    # show camera 2 RGB image and ROI
    roi_cam2 = [170, 80, 50, 100] # ROI; left, top, width, height
    [frame_rgb, frame_d] = d.get_rgbd_frame(i_frame, 1)
    cv2.rectangle(frame_rgb, tuple(roi_cam2[0:2]), (roi_cam2[0]+roi_cam2[2], roi_cam2[1]+roi_cam2[3]), (0, 0, 255), 2)
    cv2.imshow('rgb2', frame_rgb)

    # get 3D point cloud in ROI
    pc_roi1 = d.get_pc_from_rgbd(i_frame, 0, roi_cam1)
    pc_roi2 = d.get_pc_from_rgbd(i_frame, 1, roi_cam2)

    # prepare for plotting
    app=pg.QtGui.QApplication([])   
    w = gl.GLViewWidget()   
    
    # read and plot merged point cloud
    pc = d.get_mrgpc_frame(i_frame) 
    tdt.plot_pc(pc, w, 4)  

    # plot point cloud in ROIs
    tdt.plot_pc(pc_roi1, w, 5, (1,0,0,1))
    tdt.plot_pc(pc_roi2, w, 5, (1,0,0,1))

    # plot axis
    g=gl.GLAxisItem()
    w.addItem(g)

    # show the plot
    w.setCameraPosition(distance = 0.5)
    w.show()
    print('Close the window to quit.')
    pg.QtGui.QApplication.exec_()  


"""
[Python scripts for 3DTracker-FAB (www.3dtracker.org)]
Example 01: Reading images

This is a script demonstrating how to read 2D and 3D image data.

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

    i_frame = 10;   # video frame id

    # read and show RGB and depth image -------------------------------

    for i_cam in range(2):
        [frame_rgb, frame_d] = d.get_rgbd_frame(i_frame, i_cam)

        cv2.imshow('rgb - camera ' + str(i_cam+1), frame_rgb)
        frame_d_8bit = (frame_d/(2^16)).astype('uint8')
        cv2.imshow('depth - camera ' + str(i_cam+1), frame_d_8bit)
    
    print('Showing the RGB and depth images. Press any key to continue.')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # plot merged point cloud data ---------------------------
    
    # prepare for plotting
    app=pg.QtGui.QApplication([])   
    w = gl.GLViewWidget()   

    # read and plot merged point cloud data
    pc = d.get_mrgpc_frame(i_frame)
    tdt.plot_pc(pc, w, 4)  
    
    # plot axis
    g=gl.GLAxisItem()
    w.addItem(g)
    
    # show the plot
    w.setCameraPosition(distance = 0.5)
    w.show()
    print('Showing the merged point cloud. Close the window to quit.')
    pg.QtGui.QApplication.exec_()  


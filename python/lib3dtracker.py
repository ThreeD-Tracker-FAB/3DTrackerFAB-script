"""
[Python scripts for 3DTracker-FAB (www.3dtracker.org)]
The main library

This module is a set of utility functions and classes for 3DTracker-FAB.
www.3dtracker.org

Todo:
    * Functions for analyzing trace of body parts

Date last modified: 2018.10.02
"""

import glob
import numpy as np
import xml.etree.ElementTree as et
import struct
import copy
from math import cos
from math import sin

import cv2
import pyqtgraph.opengl as gl

class DataReader:
    """ This class is for utilizing 3D video data recorded 'Recorder.exe'

    Use with contextlib.closing() to open and close files safely (see Example)

    Args: 
        fname_metadata (str): file path of the metadata file (*.metadata.xml')

    Example:
        with contextlib.closing(tdt.DataReader('./data1/data1.metadata.xml')) as d:
    """

    def __init__(self, fname_metadata):
        
        n = fname_metadata.find('.metadata.xml')
        fnamebase = fname_metadata[0:n]

        # load camera intrinsic
        fname_camintrin = glob.glob(fnamebase+'.camintrin.*')
        self.camintrin = list()
        for fn in fname_camintrin:
            self.camintrin.append(CamIntrin(fn))

        # open RGBD data
        fname_rgbd = glob.glob(fnamebase+'.rgbd.frame.*')
        fname_rgbd_ts = glob.glob(fnamebase+'.rgbd.ts.*')

        self.fp_rgbd = list()
        for fn in fname_rgbd:
            self.fp_rgbd.append(open(fn, 'rb'))

        self.rgbd_ts = list()
        for fn in fname_rgbd_ts:
            self.rgbd_ts.append(np.loadtxt(fn))
        
        # open merged point cloud data
        fname_mrgpc = glob.glob(fnamebase+'.mrgpc.frame.*')
        if fname_mrgpc:
            self.fp_mrgpc = open(fname_mrgpc[0], 'rb')
            self.mrgpc_ts = np.loadtxt(glob.glob(fnamebase+'.mrgpc.ts.*')[0])
        
        # load cam transforms
        with open(fname_metadata, 'r') as fp_meta:
            buf = fp_meta.readlines()
            buf = buf[3:]
            elem = et.fromstringlist(buf)
            
            ref_cam_pos = np.zeros([3,1])
            ref_cam_rot = np.zeros([3,1])
            self.t_pc = list()

            iter = elem.iter()
            for e in iter:
                if e.tag == 'ref_cam_pos':
                    e = next(iter)
                    e = next(iter)
                    ref_cam_pos[0] = float(e.text)
                    e = next(iter)
                    ref_cam_pos[1] = float(e.text)
                    e = next(iter)
                    ref_cam_pos[2] = float(e.text)
                if e.tag == 'ref_cam_rot':
                    e = next(iter)
                    e = next(iter)
                    ref_cam_rot[0] = float(e.text)
                    e = next(iter)
                    ref_cam_rot[1] = float(e.text)
                    e = next(iter)
                    ref_cam_rot[2] = float(e.text)
                if e.tag == 'pc_transforms':
                    e = next(iter)
                    num_camera = int(e.text)
                    for i in range(num_camera):
                        e = next(iter)

                        while e.tag != 'data':
                            e = next(iter)

                        t = np.zeros([16,1])
                        for j in range(16):
                            t[j] = float(e[j].text)

                        t = t.reshape(4, 4).transpose()
                        self.t_pc.append(t)

            t = get_transformation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            t = np.dot(get_transformation(0.0, 0.0, 0.0, ref_cam_rot[0]/180.0*np.pi, 0.0, 0.0), t)
            t = np.dot(get_transformation(0.0, 0.0, 0.0, 0.0, 0.0, ref_cam_rot[2]/180.0*np.pi), t)
            t = np.dot(get_transformation(0.0, 0.0, 0.0, 0.0, ref_cam_rot[1]/180.0*np.pi, 0.0), t)
            t = np.dot(get_transformation(ref_cam_pos[0], ref_cam_pos[1], ref_cam_pos[2], 0.0, 0.0, 0.0), t)
            self.t_ref = t

    def get_rgbd_frame(self, i_frame, i_cam):
        """ The function to get RGBD data of a specified video frame captured by a specified camera

        Args:
            i_frame (int): frame id; 0 to N-1; N = frame count
            i_cam (int): camera id; 0 to M-1; M = camera count

        Returns:
            (list) list containing:

                frame_rgb (numpy array): RGB image
                frame_d (numpy array): Depth image

        """

        fp = self.fp_rgbd[i_cam]

        fp.seek(int(self.rgbd_ts[i_cam][i_frame, 1]))

        # load RGB data
        buf = fp.read(8);
        data_size = struct.unpack('Q', buf)[0]

        buf = np.asarray(bytearray(fp.read(data_size)))
        frame_rgb = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        # load depth data
        buf = fp.read(8);
        data_size = struct.unpack('Q', buf)[0]

        buf = np.asarray(bytearray(fp.read(data_size)))
        frame_d = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE or cv2.IMREAD_ANYDEPTH)
    
        return [frame_rgb, frame_d]

    def get_mrgpc_frame(self, i_frame):
        """ The function to get full 3D (merged) point cloud data of a specified video frame

        Args:
            i_frame (int): frame id; 0 to N-1; N = frame count

        Returns:
            (numpy array): point cloud data; [X,Y,Z,R,G,B,normal_x,normal_y,normal_z] x N; N = number of points  

        """
        fp = self.fp_mrgpc

        fp.seek(int(self.mrgpc_ts[i_frame, 1]))

        buf = fp.read(8);
        n_points = struct.unpack('Q', buf)[0]   # n_point: number of point in the point cloud of the frame

        pc = np.zeros((n_points,10)) # pc: point cloud data in the frame

        for i in range(n_points):
            buf = fp.read(2*3);
            xyz = struct.unpack('hhh', buf) 
            buf = fp.read(1*3);
            rgb = struct.unpack('BBB', buf) 
            buf = fp.read(1*3);
            normal = struct.unpack('bbb', buf) 

            pc[i,:] = [float(xyz[0])/2000.0, float(xyz[1])/2000.0, float(xyz[2])/2000.0, \
                       float(rgb[0])/255.0, float(rgb[1])/255.0, float(rgb[2])/255.0, 1.0, \
                       float(normal[0])/100.0, float(normal[1])/100.0, float(normal[2])/100.0 ]


        return pc

    def get_pc_from_rgbd(self, i_frame, i_cam, rect=None, shape=0):
        """ The function to get point cloud data from RGBD data of a specified video frame captured by a specified camera

        Args:
            i_frame (int): frame id; 0 to N-1; N = frame count
            i_cam (int): camera id; 0 to M-1; M = camera count
            rect (list): rectangle surrounding ROI for point cloud extraction; [left, top, width, height] in px
            shape (int): ROI shape; 0 = rectangle; 1 = Ellipse

        Returns:
            (numpy array): point cloud data; [X,Y,Z,R,G,B,normal_x,normal_y,normal_z] x N; N = number of points  

        """
        if (rect is not None) and (rect[2] == 0 or rect[3] == 0):
            return np.empty([0,7])
    
        [frame_rgb, frame_d] = self.get_rgbd_frame(i_frame, i_cam)

        pc = np.empty((frame_d.size,7))

        mask_img = np.zeros(frame_rgb.shape[0:2])
        if rect is not None:
            if shape == 0:
                cv2.rectangle(mask_img,tuple(rect[0:2]), (rect[2]+rect[0], rect[3]+rect[1]), 1.0, -1)
            elif shape == 1:
                cv2.ellipse(mask_img, (int(rect[0]+round(rect[2]/2)), int(rect[1]+round(rect[3]/2))), (int(round(rect[2]/2)), int(round(rect[3]/2))), 0, 0, 360, (1.0), -1)

        depth_in_meters = frame_d * self.camintrin[i_cam].scale

        pc_xyz = _deproject_all_pixel(depth_in_meters, self.camintrin[i_cam].depth_intrin)
        pc_xyz_in_color = _transform_cam2cam_pc(pc_xyz, self.camintrin[i_cam].depth_to_color)
        pc_px_in_color = _project_pc(pc_xyz_in_color, self.camintrin[i_cam].color_intrin)
 
        cx = np.round(pc_px_in_color[:,0]).astype(np.int)
        cy = np.round(pc_px_in_color[:,1]).astype(np.int)

        I = np.where((cx >= 0) & (cy >= 0) & (cx < self.camintrin[i_cam].color_intrin['width']) & (cy < self.camintrin[i_cam].color_intrin['height']))
        cx = cx[I]
        cy = cy[I]
        pc_xyz = np.squeeze(pc_xyz[I,:])
    
        I = np.where(pc_xyz[:,2]>0)
        cx = cx[I]
        cy = cy[I]
        pc_xyz = np.squeeze(pc_xyz[I,:])

        if (rect is not None):
            I = np.where(mask_img[cy,cx]>0)
            cx = cx[I]
            cy = cy[I]
            pc_xyz = np.squeeze(pc_xyz[I,:])

        clr = frame_rgb[cy, cx, :]
        clr = clr[:,(2,1,0)]
   
        pc_xyz = np.reshape(pc_xyz, (-1,3)) 

        pc = np.c_[pc_xyz, clr/255.0]
        pc = np.c_[pc, np.ones((pc.shape[0],1))]
        pc[:,0] = -pc[:,0]
        pc[:,1] = -pc[:,1]

        # transform according to the calibration data
        pc = transform_pc(pc, self.t_pc[i_cam]) 
        pc = transform_pc(pc, self.t_ref) 

        return pc

    def project_point_to_rgb(self, p, i_cam):
        """ The function to project a 3D point on a specified RGB camera

        Args:
            p (list): the point to project; [X, Y, Z]
            i_cam (int): camera id; 0 to M-1; M = camera count

        Returns:
            (list): Position of the projected point in the camera image; [X, Y] in px

        """
        p2 = transform_point(p, np.linalg.inv(self.t_ref))
        p2 = transform_point(p2, np.linalg.inv(self.t_pc[i_cam]))
        depth_point = [-p2[0], -p2[1], p2[2]]

        color_point = _transform_cam2cam_point(depth_point, self.camintrin[i_cam].depth_to_color)
        color_pixel = _project_point(color_point, self.camintrin[i_cam].color_intrin)
        color_pixel = np.round(color_pixel).astype(np.int)

        return color_pixel

    def close(self):

        for f in self.fp_rgbd:
            f.close()
        self.fp_rgbd = None

        if self.fp_mrgpc:
            self.fp_mrgpc.close()
            self.fp_mrgpc = None

class CamIntrin:
    """ This class is to handle camera intrinsic data easily

    """
    def __init__(self, fname_camintrin):

        with open(fname_camintrin, 'rb') as fp:

            self.depth_intrin = self.__load_intrin(fp)
            self.depth_to_color = self.__load_extrin(fp)
            self.color_intrin = self.__load_intrin(fp)
    
            buf = fp.read(4);
            self.scale = struct.unpack('f', buf)[0]

    def __load_intrin(self, fp):
    
        I = {'width':0, 'height':0, 'ppx':0, 'ppy':0, \
             'fx':0, 'fy':0, 'model':0, 'coeffs':0}

        buf = fp.read(4);
        I['width'] = struct.unpack('i', buf)[0]

        buf = fp.read(4);
        I['height'] = struct.unpack('i', buf)[0]

        buf = fp.read(4);
        I['ppx'] = struct.unpack('f', buf)[0]
    
        buf = fp.read(4);
        I['ppy'] = struct.unpack('f', buf)[0]
    
        buf = fp.read(4);
        I['fx'] = struct.unpack('f', buf)[0]
    
        buf = fp.read(4);
        I['fy'] = struct.unpack('f', buf)[0]

        buf = fp.read(4);
        I['model'] = struct.unpack('i', buf)[0]

        buf = fp.read(4*5);
        I['coeffs'] = struct.unpack('fffff', buf)

        return I

    def __load_extrin(self, fp):
    
        E = {'rotation':0, 'translation':0}
    
        buf = fp.read(4*9);
        E['rotation'] = struct.unpack('fffffffff', buf)
    
        buf = fp.read(4*3);
        E['translation'] = struct.unpack('fff', buf)

        return E

def transform_point(p, transform):
    """ The function to transform a 3D point using a transformation matrix

        Args:
            p (list): the point to transform; [X, Y, Z, ...]
            transform (numpy array): 4x4 transformation matrix 

        Returns:
            (list): the transformed point; [X, Y, Z, ...]

    """

    a = copy.deepcopy(p)
    b = transform

    p2 = copy.deepcopy(p)
    p2[0] = a[0] * b[0, 0] + a[1] * b[0, 1] + a[2] * b[0, 2] + b[0, 3]
    p2[1] = a[0] * b[1, 0] + a[1] * b[1, 1] + a[2] * b[1, 2] + b[1, 3]
    p2[2] = a[0] * b[2, 0] + a[1] * b[2, 1] + a[2] * b[2, 2] + b[2, 3]
    
    return p2

def transform_pc(pc, transform):
    """ The function to transform a 3D point cloud using a transformation matrix

        Args:
            pc (numpy array): the point cloud to transform; [X, Y, Z, ...] x N; N = num of points
            transform (numpy array): 4x4 transformation matrix 

        Returns:
            (list): the transformed point; [X, Y, Z, ...] x N

    """

    a = copy.deepcopy(pc)
    b = transform

    pc2 = copy.deepcopy(pc)
    pc2[:,0] = a[:,0] * b[0, 0] + a[:,1] * b[0, 1] + a[:,2] * b[0, 2] + b[0, 3]
    pc2[:,1] = a[:,0] * b[1, 0] + a[:,1] * b[1, 1] + a[:,2] * b[1, 2] + b[1, 3]
    pc2[:,2] = a[:,0] * b[2, 0] + a[:,1] * b[2, 1] + a[:,2] * b[2, 2] + b[2, 3]
    
    return pc2

def get_transformation(x, y, z, roll, pitch, yaw):
    """ The function to calculate a transformation matrix

        Args:
            x (float): x translation
            y (float): y translation
            z (float): z translation
            roll (float): roll
            pitch (float): pitch
            yaw (float): yaw

        Returns:
            (numpy array): the 4x4 transformation matrix

    """

    A = cos (yaw)
    B = sin (yaw)
    C  = cos (pitch)
    D  = sin (pitch)
    E = cos (roll)
    F = sin (roll)
    DE = D*E
    DF = D*F

    t = np.zeros([4,4])

    t [0, 0] = A*C
    t [0, 1] = A*DF - B*E
    t [0, 2] = B*F + A*DE
    t [0, 3] = x

    t [1, 1] = A*E + B*DF
    t [1, 2] = B*DE - A*F
    t [1, 3] = y

    t [2, 0] = -D
    t [2, 1] = C*F
    t [2, 2] = C*E
    t [2, 3] = z

    t [3, 0] = 0
    t [3, 1] = 0
    t [3, 2] = 0
    t [3, 3] = 1

    return t

def voxel_grid_filter(pc_xyz, leaf_size):
    """ The function to downsample a point cloud using a voxel grid filter

        Args:
            pc_xyz (numpy array): input point cloud, [X, Y, Z] x N; N = num of points
            leaf_size (float): leaf size (grid size) in meters

        Returns:
            (numpy array): the 4x4 transformation matrix

    """

    pc2 = np.round(pc_xyz/leaf_size)*leaf_size
    return np.unique(pc2, axis=0)

def plot_pc(pc, gl_w, point_size=3, clr=None):
    """ The function to plot a point cloud using pyqtgraph

        Args:
            pc_xyz (numpy array): input point cloud, [X, Y, Z, R, G, B, ...] x N; N = num of points
            gl_w (?): returned value of GLViewWidget()
            point_size (float): size of points
            clr (list; float x 4; 0.0-1.0): when specified, the point cloud will be shown in the specified single color

        Example:

            import pyqtgraph as pg
            import pyqtgraph.opengl as gl

            app=pg.QtGui.QApplication([])   
            w = gl.GLViewWidget()   

            tdt.plot_pc(pc, w)  

            g=gl.GLAxisItem()
            w.addItem(g)
            w.show()
    
            pg.QtGui.QApplication.exec_()  

    """

    pc_show = copy.deepcopy(pc)

    if pc_show.ndim == 1:
        pc_show = np.array([pc_show, pc_show])

    tmp = copy.deepcopy(pc_show[:,2])
    pc_show[:,2] = copy.deepcopy(pc_show[:,1])
    pc_show[:,1] = -tmp

    if clr is None:
        plt = gl.GLScatterPlotItem(pos=pc_show[:,:3], color=pc_show[:,3:7], size = point_size)
    else:
        plt = gl.GLScatterPlotItem(pos=pc_show[:,:3], color=clr, size = point_size)

    plt.setGLOptions('opaque')
    gl_w.addItem(plt)

def _deproject_pixel(pixel, depth, intrin):
    
    point = [0, 0, 0]

    x = (pixel[0] - intrin['ppx']) / intrin['fx']
    y = (pixel[1] - intrin['ppy']) / intrin['fy']

    if intrin['model'] == 1:
        r2  = x*x + y*y
        f = 1 + intrin['coeffs'][0]*r2 + intrin['coeffs'][1]*r2*r2 + intrin['coeffs'][4]*r2*r2*r2
        ux = x*f + 2*intrin['coeffs'][2]*x*y + intrin['coeffs'][3]*(r2 + 2*x*x)
        uy = y*f + 2*intrin['coeffs'][3]*x*y + intrin['coeffs'][2]*(r2 + 2*y*y)
        x = ux
        y = uy

    point[0] = depth * x
    point[1] = depth * y
    point[2] = depth

    return point

def _deproject_all_pixel(depth, intrin):
    
    x = np.tile(range(depth.shape[1]), (depth.shape[0],1))
    y = np.tile(np.array(range(depth.shape[0]), ndmin=2).T, (1, depth.shape[1]))

    x = (x - intrin['ppx']) / intrin['fx']
    y = (y - intrin['ppy']) / intrin['fy']

    if intrin['model'] == 1:
        r2  = x*x + y*y
        f = 1 + intrin['coeffs'][0]*r2 + intrin['coeffs'][1]*r2*r2 + intrin['coeffs'][4]*r2*r2*r2
        ux = x*f + 2*intrin['coeffs'][2]*x*y + intrin['coeffs'][3]*(r2 + 2*x*x)
        uy = y*f + 2*intrin['coeffs'][3]*x*y + intrin['coeffs'][2]*(r2 + 2*y*y)
        x = ux
        y = uy

    pc = np.zeros((depth.size, 3))

    pc[:,0] = (depth * x).reshape((depth.size))
    pc[:,1] = (depth * y).reshape((depth.size))
    pc[:,2] = (depth).reshape((depth.size))

    return pc

def _project_point(point, intrin):

    x = point[0] / point[2]
    y = point[1] / point[2]
    
    if intrin['model'] == 1:
        r2  = x*x + y*y
        f = 1 + intrin['coeffs'][0]*r2 + intrin['coeffs'][1]*r2*r2 + intrin['coeffs'][4]*r2*r2*r2
        x *= f
        y *= f
        dx = x + 2*intrin['coeffs'][2]*x*y + intrin['coeffs'][3]*(r2 + 2*x*x)
        dy = y + 2*intrin['coeffs'][3]*x*y + intrin['coeffs'][2]*(r2 + 2*y*y)
        x = dx
        y = dy
    
    pixel = [0, 0]
    pixel[0] = x * intrin['fx'] + intrin['ppx']
    pixel[1] = y * intrin['fy'] + intrin['ppy']

    return pixel

def _project_pc(pc, intrin):

    x = pc[:, 0] / pc[:, 2]
    y = pc[:, 1] / pc[:, 2]
    
    if intrin['model'] == 1:
        r2  = x*x + y*y
        f = 1 + intrin['coeffs'][0]*r2 + intrin['coeffs'][1]*r2*r2 + intrin['coeffs'][4]*r2*r2*r2
        x *= f
        y *= f
        dx = x + 2*intrin['coeffs'][2]*x*y + intrin['coeffs'][3]*(r2 + 2*x*x)
        dy = y + 2*intrin['coeffs'][3]*x*y + intrin['coeffs'][2]*(r2 + 2*y*y)
        x = dx
        y = dy
    
    pc_px = np.zeros((pc.shape[0], 2))
    pc_px[:,0] = x * intrin['fx'] + intrin['ppx']
    pc_px[:,1] = y * intrin['fy'] + intrin['ppy']

    return pc_px

def _transform_cam2cam_point(from_point, extrin):

    to_point = [0, 0, 0]

    to_point[0] = extrin['rotation'][0] * from_point[0] + extrin['rotation'][3] * from_point[1] + extrin['rotation'][6] * from_point[2] + extrin['translation'][0]
    to_point[1] = extrin['rotation'][1] * from_point[0] + extrin['rotation'][4] * from_point[1] + extrin['rotation'][7] * from_point[2] + extrin['translation'][1]
    to_point[2] = extrin['rotation'][2] * from_point[0] + extrin['rotation'][5] * from_point[1] + extrin['rotation'][8] * from_point[2] + extrin['translation'][2]

    return to_point

def _transform_cam2cam_pc(from_pc, extrin):

    to_pc = np.zeros(from_pc.shape)

    to_pc[:,0] = extrin['rotation'][0] * from_pc[:,0] + extrin['rotation'][3] * from_pc[:,1] + extrin['rotation'][6] * from_pc[:,2] + extrin['translation'][0]
    to_pc[:,1] = extrin['rotation'][1] * from_pc[:,0] + extrin['rotation'][4] * from_pc[:,1] + extrin['rotation'][7] * from_pc[:,2] + extrin['translation'][1]
    to_pc[:,2] = extrin['rotation'][2] * from_pc[:,0] + extrin['rotation'][5] * from_pc[:,1] + extrin['rotation'][8] * from_pc[:,2] + extrin['translation'][2]

    return to_pc


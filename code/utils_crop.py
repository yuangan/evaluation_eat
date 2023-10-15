import cv2
import numpy as np
from skimage import transform as tf


import dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords
def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def crop_image(image, resize_size = 163,crop_ratio = 0.64, bool_no_crop = False ):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)
    if len(rects) == 0 : return None , None ,False
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        if bool_no_crop :
            return image, shape, True

        (x, y, w, h) = rect_to_bb(rect)
        center_x = x + int(0.5 * w)
        center_y = y + int(0.5 * h)
        r = int(crop_ratio * h)

        new_x = center_x - r
        new_y = center_y - r
        roi = image[new_y:new_y + 2 * r, new_x:new_x + 2 * r]

        try :
            roi = cv2.resize(roi, (resize_size,resize_size), interpolation = cv2.INTER_AREA)
        except: return None,None,False

        scale =  resize_size / (2 * r)

        shape = ((shape - np.array([new_x,new_y])) * scale)

        return roi, shape, True

def crop_and_align_224(image):
    roi, landmark , ret = crop_image(image,bool_no_crop=True)
    if not ret : return False , False
    
    if  np.sum(landmark[37:39,1] - landmark[40:42,1]) < -9:
        template = np.load( 'base_68.npy')
    else:
        template = np.load( 'base_68_close.npy')

    pts2 = np.float32(template[27:45,:])
    mean_x , mean_y = pts2[:,0].mean(), pts2[:,1].mean()
    pts2[:,0] = pts2[:,0] - mean_x + 112
    pts2[:,1] = pts2[:,1] - mean_y + 112

    pts1 = np.float32(landmark[27:45,:])

    tform = tf.SimilarityTransform()
    tform.estimate( pts2, pts1)
    # dst = tf.warp(roi, tform, output_shape=(163, 163))
    dst = tf.warp(roi, tform, output_shape=(256,256))

    dst = np.array(dst * 255, dtype=np.uint8)
    # dst = dst[1:129,1:129,:]
    dst = dst[1:225,1:225,:]
    return dst , True

def crop_and_align(image):
    roi, landmark , ret = crop_image(image,bool_no_crop=True)
    if not ret : return False , False
    
    if  np.sum(landmark[37:39,1] - landmark[40:42,1]) < -9:
        template = np.load( 'base_68.npy')
    else:
        template = np.load( 'base_68_close.npy')

    pts2 = np.float32(template[27:45,:])

    pts1 = np.float32(landmark[27:45,:])

    tform = tf.SimilarityTransform()
    tform.estimate( pts2, pts1)
    dst = tf.warp(roi, tform, output_shape=(163, 163))

    dst = np.array(dst * 255, dtype=np.uint8)
    dst = dst[1:129,1:129,:]
    return dst , True
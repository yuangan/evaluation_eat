import os
import imageio
import cv2

gen_logs = open('preprocess_logs.txt','a+')

def load_frame_lis(vid_pth):
    img_lis = []
    if os.path.isdir(vid_pth) : 
        frame_lis = os.listdir(vid_pth)
        try :
            frame_lis.sort(key = lambda x : int( x.split('.')[0] ))
        except : 
            gen_logs.writelines('ERROR 1 : {} \n'.format(vid_pth))
            return None
        for frame_name in frame_lis:
            img_lis.append( cv2.cvtColor( cv2.imread( '{}/{}'.format(vid_pth,frame_name) ) , cv2.COLOR_RGB2BGR ) )
    else :
        try :
            fake_reader = imageio.get_reader( vid_pth )
        except:
            gen_logs.writelines('ERROR 0 : {} \n'.format(vid_pth))
            return None
        for im in fake_reader: img_lis.append( im )
    return img_lis
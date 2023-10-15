import argparse
import glob
import os
import random
from tkinter.tix import Tree
import tqdm
import cv2
import imageio
import numpy as np

from utils_crop import crop_and_align, crop_and_align_224
from utils import load_frame_lis

def align_crop_vids(in_vid_p,out_crop_p,size = 128):
    try :
        frames = load_frame_lis(in_vid_p)
    except:
        return False

    if frames is None : return False
    fps = 25
    sz = (size,size)
    # print(out_crop_p)
    vwriter = cv2.VideoWriter(out_crop_p,cv2.VideoWriter_fourcc('M','P','4','V'),fps,sz)
    for frame in frames:
        pro_frame = frame
        if size == 128 : pro_frame , ret = crop_and_align(frame)
        else : pro_frame , ret = crop_and_align_224(frame)
        if not ret : continue
        vwriter.write(cv2.cvtColor(pro_frame,cv2.COLOR_BGR2RGB))
    vwriter.release()

def cvt_imgs_to_vid(in_vid,out_nocrop_p,fps=25):
    vid_pth = in_vid
    img_lis = []
    sz = None
    frame_lis = glob.glob(vid_pth+'/*.jpg')
    frame_lis.extend(glob.glob(vid_pth+'/*.png'))
    #try :
    frame_lis.sort(key = lambda x : int( x.split('/')[-1].split('.')[0] ))
    #except : 
        #print('???')
        #return None
    for frame_name in frame_lis:
        img = cv2.imread('{}/{}'.format(vid_pth,frame_name.split('/')[-1]))
        if sz is None:
            sz = (img.shape[0], img.shape[1])
        img_lis.append( cv2.cvtColor( img  , cv2.COLOR_RGB2BGR ) )

    vwriter = cv2.VideoWriter(out_nocrop_p,cv2.VideoWriter_fourcc('M','P','4','V'),fps,sz)
    for frame in img_lis:
        pro_frame = frame
        vwriter.write(cv2.cvtColor(pro_frame,cv2.COLOR_BGR2RGB))
    vwriter.release()

def process_name(f,name_mode):
    if name_mode == 0 : 
        # EAMM (for dirs)
        pid , emo_lev_vid = f.split('/')[-2] , f.split('/')[-1]
        emo, _ , lev , vid = emo_lev_vid.split('_')
        emo = emo[:3]
    elif name_mode == 1 :  
        # makeittalk
        pid , emo , lev , vid , _  =  os.path.splitext( os.path.split(f)[1] )[0].split('_')
    elif name_mode == 2 : 
        # ATVG gt
        # print(os.path.splitext( os.path.split(f)[1] )[0].split('_'))
        pid , emo , lev , vid  =  os.path.splitext( os.path.split(f)[1] )[0].split('_')
    elif name_mode == 3 : 
        pid , emo , lev , vid = f.split('-')
    elif name_mode == 4 : _ , pid , emo , lev , vid  =  os.path.splitext( os.path.split(f)[1] )[0].split('_') 
    elif name_mode == 5:
        # EAMM (for processed vid)
        pid, emo, _ , lev , vid = os.path.splitext( os.path.split(f)[1] )[0].split('_') 
        emo = emo[:3]
    elif name_mode == 6:
        # PCAVS
        pid , emo , lev , vid  = f.split('/')[-2].split('_audio_')[1].split('_')
    return pid,emo,lev,vid

def get_parse():
    args = argparse.ArgumentParser('preprocess')
    args.add_argument('--save_name',type=str)
    args.add_argument('--fake_pth',type=str)
    args.add_argument('--name_mode',type=int,default=1)
    args.add_argument('--bool_only96',action='store_true')
    args.add_argument('--not_align_and_crop',action='store_true')
    args.add_argument('--ours_filter_100',action='store_true')
    args.add_argument('--need_align_crop',action='store_true')
    return args

# cmds
# gt
# python preprocess.py  --save_name gt --fake_pth '/data4/new_mead/video_test/*.mp4' --name_mode 2
# python preprocess.py  --save_name evp_gt --fake_pth '/data3/vox/vox/mead/videos_evp_25' --name_mode 2
# EAMM
# python preprocess.py --save_name EAMM --fake_pth '/data4/makeittalk_other_res/res/*/*' --name_mode 0  --need_align_crop
# python preprocess.py --save_name EAMM_filter100_2 --fake_pth '/data4/makeittalk_other_res/res/*/*' --name_mode 0 --ours_filter_100 --need_align_crop
# EADG
# python preprocess.py --save_name test-363 --fake_pth '/data4/EADG_res_othsrc/test_363/*.mp4' --name_mode 4
# python preprocess.py --save_name test-483 --fake_pth '/data4/talking_head_testing/temp_res/EADG/test_483/*.mp4' --name_mode 4
# python preprocess.py --save_name test-523 --fake_pth '/data4/talking_head_testing/temp_res/EADG/test_523/*.mp4' --name_mode 4
# python preprocess.py --save_name vt2mel25_vox_head_507 --fake_pth '/data4/talking_head_testing/temp_res/EADG/vt2mel25_vox_head_507/*.mp4' --name_mode 4
# python preprocess.py --save_name a2kp_pretrain_467_M030 --fake_pth '/data4/EADG_res/a2kp_pretrain_467_M030/*.mp4' --name_mode 4
# python preprocess.py --save_name a2kp_pretrain_467_W009 --fake_pth '/data4/EADG_res/a2kp_pretrain_467_W009/*.mp4' --name_mode 4
# python preprocess.py --save_name a2kp_pretrain_467_W015 --fake_pth '/data4/EADG_res/a2kp_pretrain_467_W015/*.mp4' --name_mode 4
# python preprocess.py --save_name a2kp_pretrain_483_filter100 --fake_pth '/data4/EADG_res/a2kp_pretrain_483/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_467_M003 --fake_pth '/data4/EADG_res/a2kp_pretrain_467_M003/*.mp4' --name_mode 4
# python preprocess.py --save_name a2kp_pretrain_489_filter100 --fake_pth '/data4/EADG_res/a2kp_pretrain_489/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_494_filter100 --fake_pth '/data4/EADG_res/a2kp_pretrain_494/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name vt2mel25_kp_2_555_filter100 --fake_pth '/data4/EADG_res/vt2mel25_kp_2_555/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name vt2mel25_2_vox_head_555_filter100 --fake_pth '/data4/EADG_res/vt2mel25_2_vox_head_555/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_2_455_filter100 --fake_pth '/data4/EADG_res/a2kp_pretrain_2_455/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_460_filter100 --fake_pth '/data4/EADG_res/a2kp_460/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_466_ori_filter100 --fake_pth '/data4/EADG_res/a2kp_pretrain_466_ori/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_471_2_filter100 --fake_pth '/data4/EADG_res/a2kp_pretrain_471_2/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_472_2_filter100 --fake_pth '/data4/EADG_res/a2kp_pretrain_472_2/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_477_2_filter100 --fake_pth '/data4/EADG_res/a2kp_pretrain_477_2/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_486_3_filter100 --fake_pth '/data4/EADG_res/a2kp_pretrain_486_3/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_490_2_filter100 --fake_pth '/data4/EADG_res/a2kp_pretrain_490_2/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_496_2_filter100 --fake_pth '/data4/EADG_res/a2kp_pretrain_496_2/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_6dof_475_filter100 --fake_pth '/data4/EADG_res/a2kp_pretrain_6dof_475/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_6dof_482_filter100 --fake_pth '/data4/EADG_res/a2kp_pretrain_6dof_482/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_poseimg_467_filter100 --fake_pth '/data4/EADG_res/a2kp_pretrain_poseimg_467/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_6dof_484_filter100 --fake_pth '/data4/EADG_res/a2kp_pretrain_6dof_484/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_6dof_483_filter100 --fake_pth '/data4/EADG_res/a2kp_pretrain_6dof_483/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_6dof_481_filter100 --fake_pth '/data4/EADG_res/a2kp_pretrain_6dof_481/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name qvt_img_538_filter100 --fake_pth '/data4/EADG_res/qvt_img_538/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name qvt_528_filter100 --fake_pth '/data4/EADG_res/qvt_528/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name poseimg_qvt_472_filter100 --fake_pth '/data4/EADG_res/poseimg_qvt_472/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name poseimg_qvt_471_filter100 --fake_pth '/data4/EADG_res/poseimg_qvt_471/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name poseimg_qvt_470_filter100 --fake_pth '/data4/EADG_res/poseimg_qvt_470/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name poseimg_qvt_473_filter100 --fake_pth '/data4/EADG_res/poseimg_qvt_473/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name qvt_img_v1_546_filter100 --fake_pth '/data4/EADG_res/qvt_img_v1_546/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posepho_494_filter100 --fake_pth '/data4/EADG_res/posepho_494/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_6dof_466_filter100 --fake_pth '/data4/EADG_res/a2kp_pretrain_6dof_466/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_6dof_477_filter100 --fake_pth '/data4/EADG_res/a2kp_pretrain_6dof_477/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_6dof_471_filter100 --fake_pth '/data4/EADG_res/a2kp_pretrain_6dof_471/*.mp4' --name_mode 4  --ours_filter_100

# python preprocess.py --save_name posedeep_464_filter100 --fake_pth '/data4/EADG_res/0819batch/posedeep_464/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posedeep_469_filter100 --fake_pth '/data4/EADG_res/0819batch/posedeep_469/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posedeep_484_filter100 --fake_pth '/data4/EADG_res/0819batch/posedeep_484/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posedeep_489_filter100 --fake_pth '/data4/EADG_res/0819batch/posedeep_489/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name poseimg_qvt_469_filter100 --fake_pth '/data4/EADG_res/0819batch/poseimg_qvt_469/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name poseimg_qvt_479_filter100 --fake_pth '/data4/EADG_res/0819batch/poseimg_qvt_479/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name poseimg_qvt_489_filter100 --fake_pth '/data4/EADG_res/0819batch/poseimg_qvt_489/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name poseimg_qvt_540_filter100 --fake_pth '/data4/EADG_res/0819batch/poseimg_qvt_540/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posepho_464_filter100 --fake_pth '/data4/EADG_res/0819batch/posepho_464/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posepho_469_filter100 --fake_pth '/data4/EADG_res/0819batch/posepho_469/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posepho_474_filter100 --fake_pth '/data4/EADG_res/0819batch/posepho_474/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posepho_479_filter100 --fake_pth '/data4/EADG_res/0819batch/posepho_479/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posepho_484_filter100 --fake_pth '/data4/EADG_res/0819batch/posepho_484/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posepho_489_filter100 --fake_pth '/data4/EADG_res/0819batch/posepho_489/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posedeep_494_filter100 --fake_pth '/data4/EADG_res/0819batch/posedeep_494/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posedeep_474_filter100 --fake_pth '/data4/EADG_res/0819batch/posedeep_474/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_6dof_471_filter100 --fake_pth '/data4/EADG_res/0819batch/a2kp_pretrain_6dof_471/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posedeep_479_filter100 --fake_pth '/data4/EADG_res/0819batch/posedeep_479/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posedeep_499_filter100 --fake_pth '/data4/EADG_res/0819batch/posedeep_499/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posepho_464_filter100 --fake_pth '/data4/EADG_res/0819batch/a2kp_pretrain_posepho_464/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posepho_469_filter100 --fake_pth '/data4/EADG_res/0819batch/a2kp_pretrain_posepho_469/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posepho_474_filter100 --fake_pth '/data4/EADG_res/0819batch/a2kp_pretrain_posepho_474/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posepho_img_494_filter100 --fake_pth '/data4/EADG_res/0819batch/posepho_img_494/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posepho_img_495_filter100 --fake_pth '/data4/EADG_res/0819batch/posepho_img_495/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posedeep_img_494_filter100 --fake_pth '/data4/EADG_res/0819batch/posedeep_img_494/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posedeep_img_495_filter100 --fake_pth '/data4/EADG_res/0819batch/posedeep_img_495/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posedeep_img_496_filter100 --fake_pth '/data4/EADG_res/0819batch/posedeep_img_496/*.mp4' --name_mode 4  --ours_filter_100

# python preprocess.py --save_name posedeep_504_filter100 --fake_pth '/data4/EADG_res/0819batch/posedeep_504/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posedeep_509_filter100 --fake_pth '/data4/EADG_res/0819batch/posedeep_509/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posedeep_514_filter100 --fake_pth '/data4/EADG_res/0819batch/posedeep_514/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posedeep_464_filter100 --fake_pth '/data4/EADG_res/0819batch/a2kp_pretrain_posedeep_464/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posedeep_469_filter100 --fake_pth '/data4/EADG_res/0819batch/a2kp_pretrain_posedeep_469/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posedeep_474_filter100 --fake_pth '/data4/EADG_res/0819batch/a2kp_pretrain_posedeep_474/*.mp4' --name_mode 4  --ours_filter_100

# python preprocess.py --save_name posedeep_img_495_all --fake_pth '/data4/EADG_res/posedeep_img_495_all/*.mp4' --name_mode 4
# python preprocess.py --save_name posepho_img_496_filter100 --fake_pth '/data4/EADG_res/0821batch/posepho_img_496/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posepho_img_497_filter100 --fake_pth '/data4/EADG_res/0821batch/posepho_img_497/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posepho_img_498_filter100 --fake_pth '/data4/EADG_res/0821batch/posepho_img_498/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posepho_img_499_filter100 --fake_pth '/data4/EADG_res/0821batch/posepho_img_499/*.mp4' --name_mode 4  --ours_filter_100

# python preprocess.py --save_name a2kp_pretrain_posedeep_img_479_filter100 --fake_pth '/data4/EADG_res/0822batch/a2kp_pretrain_posedeep_img_479/*.mp4' --name_mode 4 --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posedeep_img_480_filter100 --fake_pth '/data4/EADG_res/0822batch/a2kp_pretrain_posedeep_img_480/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posedeep_img_497_filter100 --fake_pth '/data4/EADG_res/0822batch/posedeep_img_497/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posedeep_img_498_filter100 --fake_pth '/data4/EADG_res/0822batch/posedeep_img_498/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posedeep_img_499_filter100 --fake_pth '/data4/EADG_res/0822batch/posedeep_img_499/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posedeep_img_500_filter100 --fake_pth '/data4/EADG_res/0822batch/posedeep_img_500/*.mp4' --name_mode 4  --ours_filter_100


# python preprocess.py --save_name a2kp_pretrain_posedeep_479_filter100 --fake_pth '/data4/EADG_res/0823batch/a2kp_pretrain_posedeep_479/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posedeep_484_filter100 --fake_pth '/data4/EADG_res/0823batch/a2kp_pretrain_posedeep_484/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posedeep_489_filter100 --fake_pth '/data4/EADG_res/0823batch/a2kp_pretrain_posedeep_489/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posedeep_494_filter100 --fake_pth '/data4/EADG_res/0823batch/a2kp_pretrain_posedeep_494/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posedeep_499_filter100 --fake_pth '/data4/EADG_res/0823batch/a2kp_pretrain_posedeep_499/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posepho_484_filter100 --fake_pth '/data4/EADG_res/0823batch/a2kp_pretrain_posepho_484/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posepho_489_filter100 --fake_pth '/data4/EADG_res/0823batch/a2kp_pretrain_posepho_489/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posepho_494_filter100 --fake_pth '/data4/EADG_res/0823batch/a2kp_pretrain_posepho_494/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posepho_499_filter100 --fake_pth '/data4/EADG_res/0823batch/a2kp_pretrain_posepho_499/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name poseimg_img_495_filter100 --fake_pth '/data4/EADG_res/0823batch/poseimg_img_495/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name poseimg_img_496_filter100 --fake_pth '/data4/EADG_res/0823batch/poseimg_img_496/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name poseimg_img_497_filter100 --fake_pth '/data4/EADG_res/0823batch/poseimg_img_497/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name poseimg_img_498_filter100 --fake_pth '/data4/EADG_res/0823batch/poseimg_img_498/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name poseimg_img_499_filter100 --fake_pth '/data4/EADG_res/0823batch/poseimg_img_499/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posedeep_img_500_filter100 --fake_pth '/data4/EADG_res/0823batch/posedeep_img_500/*.mp4' --name_mode 4  --ours_filter_100

# python preprocess.py --save_name posedeep_img_501_filter100 --fake_pth '/data4/EADG_res/0825batch/posedeep_img_501/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posedeep_img_502_filter100 --fake_pth '/data4/EADG_res/0825batch/posedeep_img_502/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posedeep_img_503_filter100 --fake_pth '/data4/EADG_res/0825batch/posedeep_img_503/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posedeep_img_504_filter100 --fake_pth '/data4/EADG_res/0825batch/posedeep_img_504/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posedeep_img_509_filter100 --fake_pth '/data4/EADG_res/0825batch/posedeep_img_509/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posedeep_img_514_filter100 --fake_pth '/data4/EADG_res/0825batch/posedeep_img_514/*.mp4' --name_mode 4  --ours_filter_100
        
# python preprocess.py --save_name a2kp_pretrain_posedeep_img_mead_485_filter100 --fake_pth '/data4/EADG_res/0825batch2/a2kp_pretrain_posedeep_img_mead_485/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posedeep_img_mead_489_filter100 --fake_pth '/data4/EADG_res/0825batch2/a2kp_pretrain_posedeep_img_mead_489/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posedeep_img_mead_491_filter100 --fake_pth '/data4/EADG_res/0825batch2/a2kp_pretrain_posedeep_img_mead_491/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posepho_479 --fake_pth '/data4/EADG_res/0825batch2/a2kp_pretrain_posepho_479/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posepho_504 --fake_pth '/data4/EADG_res/0825batch2/a2kp_pretrain_posepho_504/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posepho_509 --fake_pth '/data4/EADG_res/0825batch2/a2kp_pretrain_posepho_509/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posepho_514 --fake_pth '/data4/EADG_res/0825batch2/a2kp_pretrain_posepho_514/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posepho_519 --fake_pth '/data4/EADG_res/0825batch2/a2kp_pretrain_posepho_519/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posepho_524 --fake_pth '/data4/EADG_res/0825batch2/a2kp_pretrain_posepho_524/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posepho_529 --fake_pth '/data4/EADG_res/0825batch2/a2kp_pretrain_posepho_529/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posepho_534 --fake_pth '/data4/EADG_res/0825batch2/a2kp_pretrain_posepho_534/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posedeep_504_filter100 --fake_pth '/data4/EADG_res/0825batch3/a2kp_pretrain_posedeep_504/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posedeep_509_filter100 --fake_pth '/data4/EADG_res/0825batch3/a2kp_pretrain_posedeep_509/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posedeep_514_filter100 --fake_pth '/data4/EADG_res/0825batch3/a2kp_pretrain_posedeep_514/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posedeep_519_filter100 --fake_pth '/data4/EADG_res/0825batch3/a2kp_pretrain_posedeep_519/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posedeep_524_filter100 --fake_pth '/data4/EADG_res/0825batch3/a2kp_pretrain_posedeep_524/*.mp4' --name_mode 4  --ours_filter_100


# python preprocess.py --save_name a2kp_pretrain_posedeep_img_484_filter100 --fake_pth '/data4/EADG_res/0826batch/a2kp_pretrain_posedeep_img_484/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posedeep_img_486_filter100 --fake_pth '/data4/EADG_res/0826batch/a2kp_pretrain_posedeep_img_486/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posedeep_img_488_filter100 --fake_pth '/data4/EADG_res/0826batch/a2kp_pretrain_posedeep_img_488/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posedeep_img_485_filter100 --fake_pth '/data4/EADG_res/0826batch/a2kp_pretrain_posedeep_img_485/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posedeep_img_487_filter100 --fake_pth '/data4/EADG_res/0826batch/a2kp_pretrain_posedeep_img_487/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posepho_img_501_filter100 --fake_pth '/data4/EADG_res/0826batch/posepho_img_501/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posepho_img_503_filter100 --fake_pth '/data4/EADG_res/0826batch/posepho_img_503/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posepho_img_509_filter100 --fake_pth '/data4/EADG_res/0826batch/posepho_img_509/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posepho_img_500_filter100 --fake_pth '/data4/EADG_res/0826batch/posepho_img_500/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posepho_img_502_filter100 --fake_pth '/data4/EADG_res/0826batch/posepho_img_502/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posepho_img_504_filter100 --fake_pth '/data4/EADG_res/0826batch/posepho_img_504/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name posepho_img_514_filter100 --fake_pth '/data4/EADG_res/0826batch/posepho_img_514/*.mp4' --name_mode 4  --ours_filter_100

# python preprocess.py --save_name a2kp_pretrain_posedeep_img_mead_479_filter100 --fake_pth '/data4/EADG_res/0826batch2/a2kp_pretrain_posedeep_img_mead_479/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posedeep_img_mead_480_filter100 --fake_pth '/data4/EADG_res/0826batch2/a2kp_pretrain_posedeep_img_mead_480/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posedeep_img_mead_481_filter100 --fake_pth '/data4/EADG_res/0826batch2/a2kp_pretrain_posedeep_img_mead_481/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posedeep_img_mead_482_filter100 --fake_pth '/data4/EADG_res/0826batch2/a2kp_pretrain_posedeep_img_mead_482/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posedeep_img_mead_483_filter100 --fake_pth '/data4/EADG_res/0826batch2/a2kp_pretrain_posedeep_img_mead_483/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posedeep_img_only_479_filter100 --fake_pth '/data4/EADG_res/0826batch2/a2kp_pretrain_posedeep_img_only_479/*.mp4' --name_mode 4  --ours_filter_100

# python preprocess.py --save_name a2kp_pretrain_posedeep_img_481_filter100 --fake_pth '/data4/EADG_res/0826batch2/a2kp_pretrain_posedeep_img_481/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posedeep_img_482_filter100 --fake_pth '/data4/EADG_res/0826batch2/a2kp_pretrain_posedeep_img_482/*.mp4' --name_mode 4  --ours_filter_100
# python preprocess.py --save_name a2kp_pretrain_posedeep_img_483_filter100 --fake_pth '/data4/EADG_res/0826batch2/a2kp_pretrain_posedeep_img_483/*.mp4' --name_mode 4  --ours_filter_100

# python preprocess.py --save_name a2kp_posedeep_img_synconly_mead_479_filter100 --fake_pth '/data3/vox/vox/mead/result/a2kp_posedeep_img_synconly_mead_479/*.mp4' --name_mode 4  --ours_filter_100


# mit
# python preprocess.py --save_name mit --fake_pth '/data4/makeittalk_othsrc/res_vids/*.mp4'  --name_mode 1
# atvg
# python preprocess.py --save_name atvg --fake_pth '/data4/ATVG_othsrc_res/res_vids/*.mp4'  --name_mode 2 --not_align_and_crop
# fsv2v
# python preprocess.py --save_name fsv2v-453 --fake_pth '/data4/fsv2v_othsrc/osfv-ft-453/*.mp4' --name_mode 2
# python preprocess.py --save_name fsv2v-415 --fake_pth '/data4/talking_head_testing/temp_res/fsv2v/osfv-ft-415/*.mp4' --name_mode 2
# aaai22
# python preprocess.py --save_name aaai22 --fake_pth '/home/gy/gy/benchmark/AAAI22-one-shot-talking-face/test_AAAI_MEAD/*.mp4' --name_mode 4
# python preprocess.py --save_name aaai22-wopose --fake_pth '/home/gy/gy/benchmark/AAAI22-one-shot-talking-face/test_AAAI_MEAD_wopose/*.mp4' --name_mode 4
# python preprocess.py --save_name aaai22-new --fake_pth '/home/gy/gy/benchmark/AAAI22-one-shot-talking-face/test_AAAI_MEAD_gtpose/*.mp4' --name_mode 4  --need_align_crop
# audio2head
# python preprocess.py --save_name audio2head --fake_pth '/data4/talking_head_testing/temp_res/audio2head/test_0722/*.mp4'  --name_mode 2
# wav2lip
# python preprocess.py --save_name wav2lip --fake_pth '/data4/talking_head_testing/temp_res/wav2lip/test_0723/*.mp4'  --name_mode 2
# pcavs
# python preprocess.py --save_name pcavs --fake_pth '/data4/talking_head_testing/temp_res/pc_avs/temp_0726/*/G_Pose_Driven_.mp4'  --name_mode 6
# python preprocess.py --save_name pcavs_1016 --fake_pth '/data4/talking_head_testing/temp_res/pc_avs/temp_1016/*/G_Pose_Driven_.mp4'  --name_mode 6


# python preprocess.py --save_name atest_filter100 --fake_pth "/data3/vox/vox/mead/result/qvt_img_pca_sync_5_482/*.mp4" --name_mode 4  --ours_filter_100 --need_align_crop


def evp_process_videos(in_pth, out_crop_pth):
    vlis = glob.glob('{}/*.mp4'.format(in_pth))
    for vid_pth in tqdm.tqdm(vlis): 
        in_vid = vid_pth    
        pid,emo,lev,vid = process_name(in_vid,name_mode)
        if pid not in ['M003','M030','W009','W015']: continue
        sav_vname = '{}_{}_{}_{}.mp4'.format(pid,emo,lev,vid)
        # aud_name = '{}-{}-{}-{}.wav'.format(pid,emo,lev,vid)

        out_crop_p = '{}/{}'.format(out_crop_pth,sav_vname)
        in_vid_p = '{}/{}'.format(in_pth,sav_vname)

        if not os.path.exists(in_vid_p) : continue
        
        align_crop_vids(in_vid_p,out_crop_p)

if __name__ == '__main__':

    args = get_parse().parse_args()
    name_mode = args.name_mode
    save_name = args.save_name
    bool_only96 = args.bool_only96
    not_align_and_crop = args.not_align_and_crop

    out_noc_rt = '../talking_head_testing/25fps_video/no_crop/{}'.format(save_name)
    # out_noc_rt = '/data4/talking_head_testing/25fps_video_align224/no_crop/{}'.format(save_name)
    os.makedirs(out_noc_rt,exist_ok=True)
    out_fuse_rt = '../talking_head_testing/25fps_video/fuse_video_and_audio/{}'.format(save_name)
    # out_fuse_rt = '/data4/talking_head_testing/25fps_video_align224/fuse_video_and_audio/{}'.format(save_name)
    os.makedirs(out_fuse_rt,exist_ok=True)
    out_crop_pth = '../talking_head_testing/25fps_video/align_crop/{}'.format(save_name)
    # out_crop_pth = '/data4/talking_head_testing/25fps_video_align224/align_crop/{}'.format(save_name)
    os.makedirs(out_crop_pth,exist_ok=True)
    aud_rt = '../talking_head_testing/wavs_16000'
    # aud_rt = '/data4/talking_head_testing/wavs_extract_from_newmead'

    vlis = glob.glob(args.fake_pth)

    # evp_process_videos(args.fake_pth,out_crop_pth)
    # assert(0)
    
    # filter
    nvlis = []
    if args.ours_filter_100:
        fvnames = np.load('rand_sample_ours_mead100.npy',allow_pickle=True)
        for vpth in vlis:
            #print(vpth)
            pid,emo,lev,vid = process_name(vpth,name_mode)
            if pid not in ['M003','M030','W009','W015']: continue
            vname = '{}_{}_{}_{}_{}'.format(pid,pid,emo,lev,vid)
            if vname in fvnames:
                nvlis.append(vpth)
        vlis = nvlis

    
    # vlis = vlis[:2]
    print(len(vlis))

    # vnameset = []
    # for vid_pth in tqdm.tqdm(vlis):

    #     vname = vid_pth.split('/')[-1].split('.')[0]
    #     vnameset.append(vname)
    
    # vnameset = random.sample(vnameset,100)
    # print(len(vnameset))
    # print(vnameset[0])
    # np.save('rand_sample_ours_mead100.npy',vnameset)        
    # assert(0)

    # process vid to 25fps video
    for vid_pth in tqdm.tqdm(vlis):

        in_vid = vid_pth
        pid,emo,lev,vid = process_name(in_vid,name_mode)
        if pid not in ['M003','M030','W009','W015']: continue
        sav_vname = '{}_{}_{}_{}.mp4'.format(pid,emo,lev,vid)

        out_nocrop_p = '{}/{}'.format(out_noc_rt,sav_vname)

        if not os.path.exists(in_vid) : continue

        if os.path.isdir(in_vid) :
            cvt_imgs_to_vid(in_vid,out_nocrop_p,fps=25)
        else :
            adjust_cmds = 'ffmpeg -loglevel quiet -y -i {} -r 25 {}'.format(in_vid,out_nocrop_p)
            os.system(adjust_cmds)
    
    if not args.need_align_crop : exit()
    print('***************')
    # fuse
    for vid_pth in tqdm.tqdm(vlis):
        in_vid = vid_pth
        pid,emo,lev,vid = process_name(in_vid,name_mode)
        if pid not in ['M003','M030','W009','W015']: continue
        sav_vname = '{}_{}_{}_{}.mp4'.format(pid,emo,lev,vid)
        aud_name = '{}-{}-{}-{}.wav'.format(pid,emo,lev,vid)

        out_fuse_p = '{}/{}'.format(out_fuse_rt,sav_vname)
        in_vid_p = '{}/{}'.format(out_noc_rt,sav_vname)
        in_aud_p = '{}/{}'.format(aud_rt,aud_name)

        if not os.path.exists(in_vid_p) : continue
        if not os.path.exists(in_aud_p) : continue
        
        fuse_cmds = 'ffmpeg -loglevel quiet -y -i {} -i {} -vcodec copy {}'.format(in_vid_p,in_aud_p,out_fuse_p)
        os.system(fuse_cmds)

     # align and crop
    if not not_align_and_crop:
        for vid_pth in tqdm.tqdm(vlis):
            in_vid = vid_pth
            pid,emo,lev,vid = process_name(in_vid,name_mode)
            if pid not in ['M003','M030','W009','W015']: continue
            sav_vname = '{}_{}_{}_{}.mp4'.format(pid,emo,lev,vid)
            aud_name = '{}-{}-{}-{}.wav'.format(pid,emo,lev,vid)

            out_crop_p = '{}/{}'.format(out_crop_pth,sav_vname)
            in_vid_p = '{}/{}'.format(out_noc_rt,sav_vname)

            if not os.path.exists(in_vid_p) : continue
            
            align_crop_vids(in_vid_p,out_crop_p)

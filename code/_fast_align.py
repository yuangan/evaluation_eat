import glob
import os
import cv2
import argparse
import tqdm
import face_alignment
from scripts.align_68 import align_folder
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

frames_sav_rt = '../talking_head_testing/25fps_video/no_crop_frames'
vid_sav_rt    = '../talking_head_testing/25fps_video/pcavs_crop'
wav_rt        = '../talking_head_testing/wavs_extract_from_newmead'

# python _fast_align.py --name gt --gpuid 1
# python _fast_align.py --name EAMM --gpuid 1
# python _fast_align.py --name vt2mel25_vox_head_kp_lmk_567 --gpuid 1
# python _fast_align.py --name vt2mel25_vox_head_kp_555 --gpuid 1
# python _fast_align.py --name fsv2v-453 --gpuid 0
# python _fast_align.py --name vt2mel25_2_vox_head_507
# python _fast_align.py --name mit --gpuid 1
# python _fast_align.py --name pcavs --gpuid 1
# python _fast_align.py --name atvg --gpuid 0
# python _fast_align.py --name audio2head --gpuid 1
# python _fast_align.py --name wav2lip --gpuid 1
# python _fast_align.py --name aaai22 --gpuid 2

# python _fast_align.py --name a2kp_pretrain_467_M030 --gpuid 2
# python _fast_align.py --name a2kp_pretrain_467_W009 --gpuid 0
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_467_M003
# CUDA_VISIBLE_DEVICES=1 python _fast_align.py --name a2kp_pretrain_467_W015
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name a2kp_pretrain_489
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name a2kp_pretrain_bn_507

# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_483_filter100
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name a2kp_pretrain_489_filter100
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name a2kp_pretrain_494_filter100
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name vt2mel25_kp_2_555_filter100
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name vt2mel25_2_vox_head_555_filter100
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name a2kp_pretrain_2_455_filter100
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name a2kp_460_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_466_ori_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_471_2_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_472_2_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_477_2_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_486_3_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_490_2_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_496_2_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_6dof_475_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_6dof_482_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_poseimg_467_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_6dof_484_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_6dof_483_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name qvt_img_538_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name qvt_528_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name poseimg_qvt_472_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name poseimg_qvt_471_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name poseimg_qvt_470_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name poseimg_qvt_473_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name qvt_img_v1_546_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name pose_qvt_476_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name posepho_494_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_6dof_466_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_6dof_477_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_6dof_471_filter100


# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name posedeep_464_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name posedeep_484_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name posedeep_489_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name poseimg_qvt_469_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name poseimg_qvt_479_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name poseimg_qvt_489_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name poseimg_qvt_540_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name posepho_464_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name posepho_469_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name posepho_474_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name posepho_479_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name posepho_484_filter100
# CUDA_VISIBLE_DEVICES=1 python _fast_align.py --name posepho_489_filter100
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name posedeep_494_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posedeep_474_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_6dof_471_filter100
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name posedeep_479_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posedeep_499_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name a2kp_pretrain_posepho_464_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name a2kp_pretrain_posepho_469_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name a2kp_pretrain_posepho_474_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posedeep_469_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posepho_img_494_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posepho_img_495_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posedeep_img_494_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posedeep_img_495_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posedeep_img_496_filter100

# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posedeep_504_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posedeep_509_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posedeep_514_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name a2kp_pretrain_posedeep_464_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name a2kp_pretrain_posedeep_469_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name a2kp_pretrain_posedeep_474_filter100

# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posedeep_img_495_all
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posepho_img_496_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posepho_img_497_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posepho_img_498_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posepho_img_499_filter100


# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name a2kp_pretrain_posedeep_img_479_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name a2kp_pretrain_posedeep_img_480_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posedeep_img_497_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posedeep_img_498_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posedeep_img_499_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posedeep_img_500_filter100

# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name a2kp_pretrain_posedeep_479_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name a2kp_pretrain_posedeep_484_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name a2kp_pretrain_posedeep_489_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name a2kp_pretrain_posedeep_494_filter100
# CUDA_VISIBLE_DEVICES=1 python _fast_align.py --name a2kp_pretrain_posedeep_499_filter100
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name a2kp_pretrain_posepho_484_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name a2kp_pretrain_posepho_489_filter100
# CUDA_VISIBLE_DEVICES=1 python _fast_align.py --name a2kp_pretrain_posepho_494_filter100
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name a2kp_pretrain_posepho_499_filter100

# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name poseimg_img_495_filter100
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name poseimg_img_496_filter100
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name poseimg_img_497_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name poseimg_img_498_filter100
# CUDA_VISIBLE_DEVICES=1 python _fast_align.py --name poseimg_img_499_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posedeep_img_500_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posedeep_img_501_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posedeep_img_502_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posedeep_img_503_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posedeep_img_504_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posedeep_img_509_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name posedeep_img_514_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name a2kp_pretrain_posedeep_img_mead_485_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_posedeep_img_mead_489_filter100
# CUDA_VISIBLE_DEVICES=1 python _fast_align.py --name a2kp_pretrain_posedeep_img_mead_491_filter100
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name a2kp_pretrain_posedeep_504_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_posedeep_509_filter100
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name a2kp_pretrain_posedeep_514_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_posedeep_519_filter100
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name a2kp_pretrain_posedeep_524_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_posepho_479
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name a2kp_pretrain_posepho_504
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_posepho_509
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name a2kp_pretrain_posepho_514
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_posepho_519
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name a2kp_pretrain_posepho_524
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_posepho_529
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name a2kp_pretrain_posepho_534



# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_posedeep_img_484_filter100
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name a2kp_pretrain_posedeep_img_486_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_posedeep_img_488_filter100
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name a2kp_pretrain_posedeep_img_485_filter100
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name a2kp_pretrain_posedeep_img_487_filter100
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name posepho_img_501_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name posepho_img_503_filter100
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name posepho_img_509_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name posepho_img_500_filter100
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name posepho_img_502_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name posepho_img_504_filter100
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name posepho_img_514_filter100
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name a2kp_pretrain_posedeep_img_mead_479_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_posedeep_img_mead_480_filter100
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name a2kp_pretrain_posedeep_img_mead_481_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_posedeep_img_mead_482_filter100
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name a2kp_pretrain_posedeep_img_mead_483_filter100

# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_posedeep_img_only_479_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_pretrain_posedeep_img_481_filter100
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name a2kp_pretrain_posedeep_img_482_filter100
# CUDA_VISIBLE_DEVICES=3 python _fast_align.py --name a2kp_pretrain_posedeep_img_483_filter100
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name a2kp_posedeep_img_synconly_mead_479_filter100

# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name pcavs_1016
# CUDA_VISIBLE_DEVICES=2 python _fast_align.py --name aaai22-new
# CUDA_VISIBLE_DEVICES=0 python _fast_align.py --name evp_gt



def get_parser():
    parser = argparse.ArgumentParser('--')
    parser.add_argument('--name',type=str)
    parser.add_argument('--gpuid',type=int,default=0)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_parser()
    name = args.name
    gpuid = args.gpuid


    frames_sav_dir = '{}/{}'.format(frames_sav_rt,name)
    vid_sav_dir = '{}/{}'.format(vid_sav_rt,name)
    os.makedirs(frames_sav_dir,exist_ok=True)
    os.makedirs(vid_sav_dir,exist_ok=True)

    vid_pths = glob.glob('../talking_head_testing/25fps_video/no_crop/{}/*.mp4'.format(name))


    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)

    for vid_pth in tqdm.tqdm(vid_pths):
        vname = vid_pth.split('/')[-1].split('.')[0]
        sav_pth = '{}/{}.mp4'.format( vid_sav_dir , vname )

        if os.path.exists(sav_pth) : continue
        
        vid_frames_pth = '{}/{}'.format(frames_sav_dir,vname)
        os.makedirs(vid_frames_pth,exist_ok=True)
        
        # save frames
        vreader = cv2.VideoCapture(vid_pth)
        idx = 1
        while True :
            ret, frm = vreader.read()
            if not ret : break
            frame_save_path = vid_frames_pth + '/' + '%d.jpg' % idx
            cv2.imwrite( frame_save_path , frm )
            idx += 1
        
        # pcavs_crop
        align_folder(vid_frames_pth,fa=fa)
        # pcavs_align_cmd = f'CUDA_VISIBLE_DEVICES={gpuid} python scripts/align_68.py --folder_path {vid_frames_pth}'
        # print(pcavs_align_cmd)
        # os.system(pcavs_align_cmd)
        
        # gather into a video
        vid_frames_pth = vid_frames_pth + '_cropped'
        wav_pth = '{}/{}.wav'.format( wav_rt, vname )
        
        cmd = 'ffmpeg -f image2 -i {} -i {} -r 25 {} -y'.format( vid_frames_pth + '/%d.jpg' ,  wav_pth , sav_pth )
        os.system(cmd)
        # os.system(cmd)

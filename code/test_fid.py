from email.policy import default
import cv2
import numpy as np
from glob import glob
import os
import tqdm
import argparse

from utils_fid import calculate_fid, calculate_frechet_distance
from utils_crop_fid import crop_and_align
from utils import load_frame_lis

def get_pth_gt2( pid , emo , lev , vid , gtname = 'gt'  ): 
    return '../talking_head_testing/25fps_video/align_crop/{}/{}_{}_{}_{}.mp4'.format(gtname, pid , emo , lev , vid)

def get_parse():
    args = argparse.ArgumentParser('psnr_ssim')
    args.add_argument('--save_name',type=str)
    args.add_argument('--fake_pth',type=str)
    args.add_argument('--name_mode',type=int,default=6)
    args.add_argument('--only_pre_96_frames',default=False,action="store_true")
    args.add_argument('--sample_frame_num',type=int,default=5,help='sample several frames to compute FID')
    args.add_argument("--bool_multiprocessing", 
                      help="Toggle use of multiprocessing for image pre-processing. Defaults to use all cores",
                      default=False,
                      action="store_true")
    args.add_argument('--bool_crop_and_align',default=False,action='store_true')
    args.add_argument('--not_bool_crop_for_fake',default=False,action='store_true')
    args.add_argument("--batchsize", default='5',
                      help="Set batch size to use for InceptionV3 network",
                      type=int)
    args.add_argument('--gtname',type=str,default='evp_gt')
    return args

if __name__ == "__main__":
    
    args = get_parse().parse_args()

    vid_path = args.fake_pth
    save_name = args.save_name
    name_mode = args.name_mode
    only_pre_96_frames = args.only_pre_96_frames
    use_multiprocessing = args.bool_multiprocessing
    batch_size = args.batchsize
    sample_frame_num = args.sample_frame_num
    bool_crop_and_align = args.bool_crop_and_align
    not_bool_crop_for_fake = args.not_bool_crop_for_fake



    vlis = glob(vid_path)
    fake_acts , gt_acts = [] , []
    fg = open('results/{}.txt'.format(save_name), 'w+')

    
    '''compute'''
    iter = 0
    for dat in tqdm.tqdm(vlis):
        iter += 1    
        f = dat

        # ~ get path
        if name_mode == 0 : 
            # EAMM
            pid , emo_lev_vid = f.split('/')[-2] , f.split('/')[-1]
            emo, _ , lev , vid = emo_lev_vid.split('_')
            emo = emo[:3]
        elif name_mode == 1 :  
            # makeittalk
            pid , emo , lev , vid , _  =  os.path.splitext( os.path.split(f)[1] )[0].split('_')
        elif name_mode == 2 : 
            # ATVG
            pid , emo , lev , vid  =  os.path.splitext( os.path.split(f)[1] )[0].split('_')
        elif name_mode == 3 : pid , emo , lev , vid = dat.split('-')
        elif name_mode == 4 : _, pid , emo , lev , vid  =  os.path.splitext( os.path.split(f)[1] )[0].split('_')
        elif name_mode == 5:
            # EAMM (for processed vid)
            pid, emo, _ , lev , vid = os.path.splitext( os.path.split(f)[1] )[0].split('_') 
            emo = emo[:3]
        elif name_mode == 6:
            pid, emo, lev , vid = os.path.split(f)[1].split('.')[0].split('_')
    
        gt_f = get_pth_gt2( pid , emo , lev , vid , gtname = args.gtname )

        if not os.path.exists(f) : 
            fg.write(f'fake {f}: not exists! \n')
            fg.flush()
            continue
        if not os.path.exists(gt_f) : 
            fg.write(f'gt {gt_f}: not exists! \n')
            fg.flush()
            continue
        
        #  ~ load video
        fake_img_lis = load_frame_lis(f)
        if fake_img_lis is None:
            fg.write(f'fake {f}: sort error! \n')
            fg.flush()
            continue
        gt_img_lis = load_frame_lis(gt_f)
        if gt_img_lis is None:
            fg.write(f'gt {gt_f}: sort error! \n')
            fg.flush()
            continue
        
        # ~  align frames
        if only_pre_96_frames:
            # only consider pre 96 frames. notion: frames must be aligned betweem fake and gt.
            length = min( 96 , min( len( fake_img_lis ) , len( gt_img_lis )  ) )
            fake_img_lis = np.array(fake_img_lis[:length])
            gt_img_lis = np.array(gt_img_lis[:length])
        else :
            # all frames. 
            length = min( len(fake_img_lis) , len(gt_img_lis) )
            fake_lis_id = np.linspace(0,len(fake_img_lis),length,False).astype(np.int32).tolist()
            gt_lis_id = np.linspace(0,len(gt_img_lis),length,False).astype(np.int32).tolist()
            fake_img_lis = np.array(fake_img_lis)[fake_lis_id]
            gt_img_lis = np.array(gt_img_lis)[gt_lis_id]
        
        # ~  crop
        if bool_crop_and_align:
            croped_fake_img_lis, croped_gt_img_lis = [] , []
            for frame_id in range( length ):
                gt_img , fake_img = gt_img_lis[ frame_id ] , fake_img_lis[ frame_id ]
                gt_img  , ret  = crop_and_align(gt_img  )
                if not ret : continue
                if not not_bool_crop_for_fake:
                    fake_img  , ret  = crop_and_align(fake_img  )
                    if not ret : continue
                croped_fake_img_lis.append(fake_img)
                croped_gt_img_lis.append(gt_img)
            length = len( croped_gt_img_lis )
            if length == 0 :
                fg.write(f'do not detect face in {f} \n')
                fg.flush()
                continue
            fake_img_lis, gt_img_lis = np.array( croped_fake_img_lis ) , np.array(  croped_gt_img_lis )    

        # ~  sample several frames for each video
        samp_idxs = np.linspace(0,length,min(length,sample_frame_num),False).astype(np.int32).tolist()
        fake_img_lis = np.array(fake_img_lis)[samp_idxs]
        gt_img_lis = np.array(gt_img_lis)[samp_idxs]

        # test fid
        for idx , im in enumerate(fake_img_lis):
            im = im[:, :, ::-1] # Convert from BGR to RGB
            fake_img_lis[idx] = im
        for idx , im in enumerate(gt_img_lis):
            im = im[:, :, ::-1] # Convert from BGR to RGB
            gt_img_lis[idx] = im
        
        fake_act , gt_act = calculate_fid( fake_img_lis , gt_img_lis,use_multiprocessing, batch_size, return_act=True  )
        fake_acts.append(fake_act)
        gt_acts.append(gt_act)
        # print(fake_act.shape)
        # print(gt_act.shape)
        fg.write(f'processing {f} \n')
        fg.flush()

    fake_acts , gt_acts = np.concatenate(fake_acts,axis=0) , np.concatenate(gt_acts,axis=0)


    fake_mu = np.mean(fake_acts, axis=0)
    fake_sigma = np.cov(fake_acts, rowvar=False)
    gt_mu = np.mean(gt_acts, axis=0)
    gt_sigma = np.cov(gt_acts, rowvar=False)

    fid = calculate_frechet_distance(fake_mu, fake_sigma, gt_mu, gt_sigma)
    fg.write(f'fid: {fid}')







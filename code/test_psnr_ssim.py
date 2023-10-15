
import argparse
from glob import glob
from tracemalloc import start

from torch import save
from utils_crop_psnr import crop_and_align
import tqdm
import os
import imageio
import cv2
import numpy as np

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def get_pth_gt2( pid , emo , lev , vid , gtname = 'gt'  ): 
    return '../talking_head_testing/25fps_video/align_crop/{}/{}_{}_{}_{}.mp4'.format(gtname, pid , emo , lev , vid)

def str2bool(str):
    return True if str == 'True' else False

def get_parse():
    args = argparse.ArgumentParser('psnr_ssim')
    args.add_argument('--save_name',type=str)
    args.add_argument('--fake_pth',type=str)
    args.add_argument('--name_mode',type=int,default=6)
    args.add_argument('--bool_crop_and_align',type=str2bool,default=True, help = 'whether need align and crop!')
    args.add_argument('--bool_crop_for_fake',type=str2bool,default=True, help = 'whether need align and crop for fake videos. always False when bool_crop_and_align is False.')
    args.add_argument('--gtname',type=str,default='evp_gt')
    args.add_argument('--start_frame_id',type=int,default=0)
    args.add_argument('--ours_filter_100',action='store_true')
    return args

if __name__ == '__main__':
    args = get_parse().parse_args()

    vid_path = args.fake_pth
    save_name = args.save_name
    name_mode = args.name_mode
    bool_crop_and_align = args.bool_crop_and_align
    bool_crop_for_fake = args.bool_crop_for_fake

    print(vid_path)
    vlis = glob(vid_path)

    if args.ours_filter_100:
        nvlis = []
        vnames = np.load('../talking_head_testing/code/rand_sample_ours_mead100.npy',allow_pickle=True)
        for vid_pth in vlis:
            in_vid = vid_pth
            vname = in_vid.split('/')[-1].split('.')[0]
            vvvname = vname.split('_')[0] + '_' + vname
            if vvvname in vnames:
                nvlis.append(vid_pth)
        vlis = nvlis
    print(len(vlis))


    res_psnr , res_ssim = [] , []
    fg = open('result_psnr/{}.txt'.format(save_name), 'w+')

    iter = 0
    for dat in tqdm.tqdm(vlis):
        iter += 1    
        f = dat

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
        elif name_mode == 4 : _ , pid , emo , lev , vid = os.path.splitext( os.path.split(f)[1] )[0].split('_')
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

        gt_reader = imageio.get_reader( gt_f )
        
        gt_img_lis, fake_img_lis = [], []
        for im in gt_reader: gt_img_lis.append( im )
        if os.path.isdir(f) : 
            frame_lis = os.listdir(f)
            try :
                frame_lis.sort(key = lambda x : int( x.split('.')[0] ))
            except : continue
            for frame_name in frame_lis:
                fake_img_lis.append( cv2.cvtColor( cv2.imread( '{}/{}'.format(f,frame_name) ) , cv2.COLOR_RGB2BGR ) )
        else :
            fake_reader = imageio.get_reader( f )
            for im in fake_reader: fake_img_lis.append( im )
        
        # consider all frames. TODO if need only consider first 96 frames, please reference code_fid/main.py.
        length = min( len(fake_img_lis) , len(gt_img_lis) )
        fake_lis_id = np.linspace(0,len(fake_img_lis),length,False).astype(np.int32).tolist()
        gt_lis_id = np.linspace(0,len(gt_img_lis),length,False).astype(np.int32).tolist()
        fake_img_lis = np.array(fake_img_lis)[fake_lis_id]
        gt_img_lis = np.array(gt_img_lis)[gt_lis_id]

        
        vid_psnr = []
        vid_ssim = []
        new_x , new_y , r = None , None , None
        for id in range( min( length - 1 , args.start_frame_id ) ,length):

            gt_img, fake_img = gt_img_lis[id] , fake_img_lis[id]
            
            if bool_crop_and_align:
                gt_img  , ret  = crop_and_align(gt_img  )
                if not ret : continue
                if bool_crop_for_fake:
                    fake_img  , ret  = crop_and_align(fake_img  )
                    if not ret : continue

            #img_psnr = compare_psnr( gt_img , fake_img , data_range=255)
            img_psnr = compare_psnr( gt_img , fake_img)
            img_ssim = compare_ssim(gt_img , fake_img, multichannel=True)
            
            vid_psnr.append(img_psnr)
            vid_ssim.append(img_ssim)
            res_psnr.append(img_psnr)
            res_ssim.append(img_ssim)


        if len(vid_ssim) > 0 and len(vid_psnr) > 0 : fg.write(f'{f} | psnr: {sum(vid_psnr) / len(vid_psnr)} | ssim: {sum(vid_ssim) / len(vid_ssim)}  \n')
        if iter % 10 == 0 and len(res_psnr) > 0 and len(res_ssim) > 0 : fg.write(f'avg: | psnr: {sum(res_psnr) / len(res_psnr)} | ssim: {sum(res_ssim) / len(res_ssim)}  \n')
        fg.flush()
        
    fg.write(f'total: | psnr: {sum(res_psnr) / len(res_psnr)} | ssim: {sum(res_ssim) / len(res_ssim)}  \n')


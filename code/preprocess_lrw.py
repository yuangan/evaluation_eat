import argparse
import glob
import os
import tqdm
import cv2
import imageio
import numpy as np
import random

from utils_crop import crop_and_align, crop_and_align_224
from utils import load_frame_lis

def align_crop_vids(in_vid_p,out_crop_p,size = 128):
    frames = load_frame_lis(in_vid_p)
    if frames is None : return False
    fps = 25
    sz = (size,size)
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
    try:
        if name_mode == 0 : 
            word , vid , _ , _ = f.split('/')[-1].split('_')
        elif name_mode == 1:
            word , vid = f.split('/')[-1].split('.')[0].split('_')
        elif name_mode == 2:
            word , vid, _ = f.split('/')[-1].split('.')[0].split('_')
        elif name_mode == 3:
            word, vid = f.split('/')[-2].split('_audio_')[1].split('_')
        elif name_mode == 4:
            _, word, vid = f.split('/')[-1].split('.')[0].split('_')
        elif name_mode == 5:
            # PCAVS
            word, vid  = f.split('/')[-2].split('_audio_')[1].split('_')
    except:
        return None, None
    return word , vid
def get_parse():
    args = argparse.ArgumentParser('preprocess')
    args.add_argument('--save_name',type=str)
    args.add_argument('--fake_pth',type=str)
    args.add_argument('--audio_path',type=str,default='../talking_head_testing/lipread_test_wav')
    args.add_argument('--name_mode',type=int,default=1)
    args.add_argument('--bool_only96',action='store_true')
    args.add_argument('--not_align_and_crop',action='store_true')
    args.add_argument('--not_frames',action='store_true')
    args.add_argument('--not_fuseaudio',action='store_true')
    args.add_argument('--ours_filter_100',action='store_true')
    args.add_argument('--need_align_crop',action='store_true')
    return args

# cmds
# gt
# python preprocess.py --save_name gt --fake_pth '/data3/lipread_test_25/video/*.mp4' --name_mode 1
# EAMM
# python preprocess.py --save_name EAMM_filter100 --fake_pth '/data4/talking_head_testing_lrw/temp_res/EAMM/all/*/*.mp4' --name_mode 1 --ours_filter_100
# mit
# python preprocess.py --save_name makeittalk --fake_pth '/data4/talking_head_testing_lrw/temp_res/makeittalk/0726/*.mp4' --name_mode 2
# audio2head
# python preprocess.py --save_name audio2head --fake_pth '/data4/talking_head_testing_lrw/temp_res/audio2head/test_0802/*.mp4' --name_mode 1
# atvg
# python preprocess.py --save_name atvg --fake_pth '/data4/talking_head_testing_lrw/temp_res/atvg/*.mp4' --name_mode 1
# wav2lip
# python preprocess.py --save_name wav2lip --fake_pth '/data4/talking_head_testing_lrw/temp_res/wav2lip/test_0726/*.mp4' --name_mode 1
# EAMM
# python preprocess.py --save_name EAMM --fake_pth '/data4/talking_head_testing_lrw/temp_res/EAMM/0814/*/*.mp4' --name_mode 1
# AAAI22
# python preprocess.py --save_name AAAI22 --fake_pth '/home/gy/gy/benchmark/AAAI22-one-shot-talking-face/test_AAAI_LRW_gtpose/*' --name_mode 4 --need_align_crop
# pcavs
# python preprocess.py --save_name pcavs --fake_pth '/data4/talking_head_testing_lrw/temp_res/pcavs/1102/*/G_Pose_Driven_.mp4' --name_mode 5 --need_align_crop


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
    aud_rt = args.audio_path
    # aud_rt = '/data4/talking_head_testing/wavs_extract_from_newmead'

    if '*.mp4' not in args.fake_pth:
        vlis = glob.glob( os.path.join( args.fake_pth, '*.mp4' ) )
    else :
        vlis = glob.glob(args.fake_pth)
    
    # vnames = []
    # for vid_pth in tqdm.tqdm(vlis):
    #     in_vid = vid_pth
    #     word , vid = process_name(in_vid,name_mode)
    #     vnames.append('{}_{}'.format(word,vid))
    # vnames = random.sample(vnames,100)
    # np.save('random_filter100.npy',vnames)
    # assert(0)

    if args.ours_filter_100:
        nvlis = []
        vnames = np.load('rand_sample_lrw_filter100.npy',allow_pickle=True)
        for vid_pth in vlis:
            in_vid = vid_pth
            word , vid = process_name(in_vid,name_mode)
            if word is None : continue
            if '{}_{}'.format(word,vid) in vnames:
                nvlis.append(vid_pth)
        vlis = nvlis

    # vlis = vlis[:2]
    print(len(vlis))

    # process vid to 25fps video
    if not args.not_fuseaudio:
        for vid_pth in tqdm.tqdm(vlis):
            in_vid = vid_pth
            word , vid = process_name(in_vid,name_mode)
            if word is None : continue
            sav_vname = '{}_{}.mp4'.format(word , vid)

            out_nocrop_p = '{}/{}'.format(out_noc_rt,sav_vname)

            if not os.path.exists(in_vid) : continue

            if os.path.isdir(in_vid) :
                cvt_imgs_to_vid(in_vid,out_nocrop_p,fps=25)
            else :
                adjust_cmds = 'ffmpeg -loglevel quiet -y -i {} -r 25 {}'.format(in_vid,out_nocrop_p)
                os.system(adjust_cmds)
    
    if not args.need_align_crop : exit()

    # fuse
    if not args.not_frames:
        for vid_pth in tqdm.tqdm(vlis):
            in_vid = vid_pth
            word , vid = process_name(in_vid,name_mode)
            if word is None : continue
            sav_vname = '{}_{}.mp4'.format(word , vid)
            aud_name = '{}_{}.wav'.format(word , vid)

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
            word , vid = process_name(in_vid,name_mode)
            if word is None : continue
            sav_vname = '{}_{}.mp4'.format(word , vid)
            aud_name = '{}_{}.wav'.format(word , vid)

            out_crop_p = '{}/{}'.format(out_crop_pth,sav_vname)
            in_vid_p = '{}/{}'.format(out_noc_rt,sav_vname)

            if not os.path.exists(in_vid_p) : continue
            
            align_crop_vids(in_vid_p,out_crop_p)

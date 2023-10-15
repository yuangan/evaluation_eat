import argparse
import glob
import os
import tqdm
import cv2
import imageio

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
    frame_lis = os.listdir(vid_pth)
    try :
        frame_lis.sort(key = lambda x : int( x.split('.')[0] ))
    except : return None
    for frame_name in frame_lis:
        img = cv2.imread('{}/{}'.format(vid_pth,frame_name))
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
        pid , emo , lev , vid  =  os.path.splitext( os.path.split(f)[1] )[0].split('_')
    elif name_mode == 3 : 
        pid , emo , lev , vid = f.split('-')
    elif name_mode == 4 : _ , pid , emo , lev , vid  =  os.path.splitext( os.path.split(f)[1] )[0].split('_') 
    elif name_mode == 5:
        # EAMM (for processed vid)
        pid, emo, _ , lev , vid = os.path.splitext( os.path.split(f)[1] )[0].split('_') 
        emo = emo[:3]
    return pid,emo,lev,vid

def get_parse():
    args = argparse.ArgumentParser('preprocess')
    args.add_argument('--save_name',type=str)
    args.add_argument('--fake_pth',type=str)
    args.add_argument('--name_mode',type=int,default=1)
    args.add_argument('--bool_only96',action='store_true')
    args.add_argument('--not_align_and_crop',action='store_true')
    return args

# cmds
# gt
# python preprocess_224_align.py  --save_name gt --fake_pth '/data4/new_mead/video_test/*.mp4' --name_mode 2
# EAMM
# python preprocess_224_align.py --save_name EAMM --fake_pth '/data4/makeittalk_other_res/res/*/*' --name_mode 0
# EADG
# python preprocess_224_align.py --save_name test-363 --fake_pth '/data4/EADG_res_othsrc/test_363/*.mp4' --name_mode 4
# python preprocess_224_align.py --save_name test-483 --fake_pth '/data4/talking_head_testing/temp_res/EADG/test_483/*.mp4' --name_mode 4
# python preprocess_224_align.py --save_name test-523 --fake_pth '/data4/talking_head_testing/temp_res/EADG/test_523/*.mp4' --name_mode 4
# python preprocess_224_align.py --save_name vt2mel25_vox_head_507 --fake_pth '/data4/talking_head_testing/temp_res/EADG/vt2mel25_vox_head_507/*.mp4' --name_mode 4

# mit
# python preprocess_224_align.py --save_name mit --fake_pth '/data4/makeittalk_othsrc/res_vids/*.mp4'  --name_mode 1
# atvg
# python preprocess_224_align.py --save_name atvg --fake_pth '/data4/ATVG_othsrc_res/res_vids/*.mp4'  --name_mode 2 --not_align_and_crop
# fsv2v
# python preprocess_224_align.py --save_name fsv2v-453 --fake_pth '/data4/fsv2v_othsrc/osfv-ft-453/*.mp4' --name_mode 2
# python preprocess_224_align.py --save_name fsv2v-415 --fake_pth '/data4/talking_head_testing/temp_res/fsv2v/osfv-ft-415/*.mp4' --name_mode 2
# aaai22
# python preprocess_224_align.py --save_name aaai22 --fake_pth '/home/gy/gy/benchmark/AAAI22-one-shot-talking-face/test_AAAI_MEAD/*.mp4' --name_mode 4
# python preprocess_224_align.py --save_name aaai22-wopose --fake_pth '/home/gy/gy/benchmark/AAAI22-one-shot-talking-face/test_AAAI_MEAD_wopose/*.mp4' --name_mode 4
# audio2head
# python preprocess_224_align.py --save_name audio2head --fake_pth '/data4/talking_head_testing/temp_res/audio2head/test_0722/*.mp4'  --name_mode 2




if __name__ == '__main__':

    args = get_parse().parse_args()
    name_mode = args.name_mode
    save_name = args.save_name
    bool_only96 = args.bool_only96
    not_align_and_crop = args.not_align_and_crop

    # out_noc_rt = '/data4/talking_head_testing/25fps_video/no_crop/{}'.format(save_name)
    out_noc_rt = '/data4/talking_head_testing/25fps_video_align224/no_crop/{}'.format(save_name)
    os.makedirs(out_noc_rt,exist_ok=True)
    # out_fuse_rt = '/data4/talking_head_testing/25fps_video/fuse_video_and_audio/{}'.format(save_name)
    out_fuse_rt = '/data4/talking_head_testing/25fps_video_align224/fuse_video_and_audio/{}'.format(save_name)
    os.makedirs(out_fuse_rt,exist_ok=True)
    # out_crop_pth = '/data4/talking_head_testing/25fps_video/align_crop/{}'.format(save_name)
    out_crop_pth = '/data4/talking_head_testing/25fps_video_align224/align_crop/{}'.format(save_name)
    os.makedirs(out_crop_pth,exist_ok=True)
    # aud_rt = '/data4/talking_head_testing/wavs_16000'
    aud_rt = '/data4/talking_head_testing/wavs_extract_from_newmead'

    vlis = glob.glob(args.fake_pth)
    # vlis = vlis[:2]
    print(len(vlis))

    # process vid to 25fps video
    for vid_pth in tqdm.tqdm(vlis):
        in_vid = vid_pth
        pid,emo,lev,vid = process_name(in_vid,name_mode)
        sav_vname = '{}_{}_{}_{}.mp4'.format(pid,emo,lev,vid)

        out_nocrop_p = '{}/{}'.format(out_noc_rt,sav_vname)

        if not os.path.exists(in_vid) : continue

        if os.path.isdir(in_vid) :
            cvt_imgs_to_vid(in_vid,out_nocrop_p,fps=25)
        else :
            adjust_cmds = 'ffmpeg -loglevel quiet -y -i {} -r 25 {}'.format(in_vid,out_nocrop_p)
            os.system(adjust_cmds)

    # fuse
    for vid_pth in tqdm.tqdm(vlis):
        in_vid = vid_pth
        pid,emo,lev,vid = process_name(in_vid,name_mode)
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
            sav_vname = '{}_{}_{}_{}.mp4'.format(pid,emo,lev,vid)
            aud_name = '{}-{}-{}-{}.wav'.format(pid,emo,lev,vid)

            out_crop_p = '{}/{}'.format(out_crop_pth,sav_vname)
            in_vid_p = '{}/{}'.format(out_noc_rt,sav_vname)

            if not os.path.exists(in_vid_p) : continue
            
            align_crop_vids(in_vid_p,out_crop_p,size = 224)

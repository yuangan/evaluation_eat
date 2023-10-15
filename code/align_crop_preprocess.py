from tqdm import tqdm
import os
import argparse
import glob
import cv2
from utils import load_frame_lis
from utils_crop import crop_and_align

def process_vid(vid_pth,save_pth,not_align_crop = False):
    print(vid_pth,save_pth)
    frames = load_frame_lis(vid_pth)
    if frames is None : return False
    fps = 25
    if not_align_crop: sz = (256,256)
    else : sz = (128,128)
    vwriter = cv2.VideoWriter(save_pth,cv2.VideoWriter_fourcc('M','P','4','V'),fps,sz)
    for frame in frames:
        pro_frame = frame
        if not not_align_crop :
            pro_frame , ret = crop_and_align(frame)
            if not ret : continue
        vwriter.write(cv2.cvtColor(pro_frame,cv2.COLOR_BGR2RGB))
    vwriter.release()

# cmds
# python align_crop_preprocess.py --fake_pth '/data4/EADG_res/test_363/*.mp4'  --sav_rt /data4/talking_head_testing/align_crop_res_25fps/EADG/363 
# python align_crop_preprocess.py --fake_pth '/data4/EADG_res_othsrc/test_363/*.mp4'  --sav_rt /data4/talking_head_testing/align_crop_res_25fps/EADG/363-all 
# python align_crop_preprocess.py --fake_pth '/data4/makeittalk_other_res/res/*/*'  --sav_rt /data4/talking_head_testing/align_crop_res_25fps/EAMM/theirs --EAMM_name_mode  
# python align_crop_preprocess.py --fake_pth '/data4/makeittalk_othsrc/res_vids/*.mp4' --sav_rt /data4/talking_head_testing/align_crop_res_25fps/makeittalk/mit 
# python align_crop_preprocess.py --fake_pth '/data4/ATVG_othsrc_res/res_vids/*.mp4' --sav_rt /data4/talking_head_testing/align_crop_res_25fps/atvg/atvg 
# python align_crop_preprocess.py --fake_pth '/data4/new_mead/video_test/*.mp4' --sav_rt /data4/talking_head_testing/align_crop_res_25fps/gt 
# fsv2v
# python align_crop_preprocess.py --fake_pth '/data4/fsv2v_othsrc/osfv-ft-453/*.mp4' --sav_rt /data4/talking_head_testing/align_crop_res_25fps/fsv2v/453 
# python align_crop_preprocess.py --fake_pth '/data4/talking_head_testing/temp_res/fsv2v/osfv-ft-415/*.mp4' --sav_rt /data4/talking_head_testing/align_crop_res_25fps/fsv2v/415 
# aaai22
# python align_crop_preprocess.py --fake_pth '/home/gy/gy/benchmark/AAAI22-one-shot-talking-face/test_AAAI_MEAD/*.mp4' --sav_rt /data4/talking_head_testing/align_crop_res_25fps/aaai22/basic
# python align_crop_preprocess.py --fake_pth '/home/gy/gy/benchmark/AAAI22-one-shot-talking-face/test_AAAI_MEAD_wopose/*.mp4' --sav_rt /data4/talking_head_testing/align_crop_res_25fps/aaai22/wopose


# python align_crop_preprocess.py --fake_pth '/data4/EADG_res/test_363/*.mp4'  --sav_rt /data4/talking_head_testing/vid_25fps/EADG/363 --not_align_crop 
# python align_crop_preprocess.py --fake_pth '/data4/makeittalk_other_res/res/*/*'  --sav_rt /data4/talking_head_testing/vid_25fps/EAMM/theirs --not_align_crop --EAMM_name_mode 
# python align_crop_preprocess.py --fake_pth '/data4/makeittalk_othsrc/res_vids/*.mp4' --sav_rt /data4/talking_head_testing/vid_25fps/makeittalk/mit --not_align_crop 
# python align_crop_preprocess.py --fake_pth '/data4/ATVG_othsrc_res/res_vids/*.mp4' --sav_rt /data4/talking_head_testing/vid_25fps/atvg/atvg --not_align_crop 
# python align_crop_preprocess.py --fake_pth '/data4/new_mead/video_test/*.mp4' --sav_rt /data4/talking_head_testing/vid_25fps/gt --not_align_crop 
# fsv2v
# python align_crop_preprocess.py --fake_pth '/data4/fsv2v_othsrc/osfv-ft-453/*.mp4' --sav_rt /data4/talking_head_testing/vid_25fps/fsv2v/453 --not_align_crop 
# python align_crop_preprocess.py --fake_pth '/data4/talking_head_testing/temp_res/fsv2v/osfv-ft-415/*.mp4' --sav_rt /data4/talking_head_testing/vid_25fps/fsv2v/415 --not_align_crop
# aaai22
# python align_crop_preprocess.py --fake_pth '/home/gy/gy/benchmark/AAAI22-one-shot-talking-face/test_AAAI_MEAD/*.mp4' --sav_rt /data4/talking_head_testing/vid_25fps/aaai22/basic --not_align_crop
# python align_crop_preprocess.py --fake_pth '/home/gy/gy/benchmark/AAAI22-one-shot-talking-face/test_AAAI_MEAD_wopose/*.mp4' --sav_rt /data4/talking_head_testing/vid_25fps/aaai22/wopose --not_align_crop

def get_parse():
    args = argparse.ArgumentParser('align and crop') 
    args.add_argument('--fake_pth',type=str)
    args.add_argument('--sav_rt',type=str)
    args.add_argument('--not_align_crop',action='store_true')
    args.add_argument('--EAMM_name_mode',action='store_true')
    return args
    
        
if __name__ == '__main__':
    args = get_parse().parse_args()
    sav_rt = args.sav_rt
    not_align_crop = args.not_align_crop
    os.makedirs(sav_rt,exist_ok=True)
    vid_pths = glob.glob(args.fake_pth)
    for vid_pth in tqdm(vid_pths):
        vname = vid_pth.split('/')[-1]
        if args.EAMM_name_mode:
            vname = vid_pth.split('/')[-2] + '_' + vname.split('.')[0] + '.mp4'
        else :
            vname = vname.split('.')[0] + '.mp4' 
        sav_pth = '{}/{}'.format(sav_rt,vname)
        process_vid(vid_pth,sav_pth,not_align_crop)


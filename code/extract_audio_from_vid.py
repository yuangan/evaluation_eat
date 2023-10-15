import glob
import tqdm
import os

# ffmpeg -i {} {}

vid_pths = glob.glob('/data4/talking_head_testing/25fps_video/cvt_by_ffmpeg_directly/gt/*.mp4')

wav_out_rt = '/data4/talking_head_testing/wavs_extract_from_newmead'
os.makedirs(wav_out_rt,exist_ok=True)

for vid_pth in tqdm.tqdm(vid_pths):
    vname = vid_pth.split('/')[-1]
    wname = vname.replace('.mp4','.wav')
    out_pth = '{}/{}'.format(wav_out_rt,wname)
    cmd = 'ffmpeg -loglevel quiet -i {} {}'.format(vid_pth,out_pth)
    os.system(cmd)

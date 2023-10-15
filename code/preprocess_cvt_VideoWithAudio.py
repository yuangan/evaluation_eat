import glob
import os
import tqdm

# ffmpeg -i {mp} -filter:v fps=25 -ac 1 -ar 16000 {outfile}


vpth = '/data4/new_mead/video_test/*.mp4'
sav_name = 'gt'

out_rt = '/data4/talking_head_testing/25fps_video/cvt_by_ffmpeg_directly/{}'.format(sav_name)
os.makedirs(out_rt,exist_ok=True)

if __name__ == '__main__':
    vid_pths = glob.glob(vpth)
    for vid_pth in tqdm.tqdm(vid_pths):
        vname = vid_pth.split('/')[-1]
        out_pth = '{}/{}'.format(out_rt,vname) 
        cmd = 'ffmpeg -loglevel quiet -i {} -filter:v fps=25 -ac 1 -ar 16000 {}'.format(vid_pth,out_pth)
        os.system(cmd)
        # assert(0)
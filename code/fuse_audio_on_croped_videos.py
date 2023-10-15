import os
import glob
import argparse
import tqdm

def get_args():
    parser = argparse.ArgumentParser('fuse audio and video')
    parser.add_argument('--name',type=str)
    return parser.parse_args()


# vid_src = 'align_crop'
# vid_target = 'align_crop_audio'
# wav_rt = '/data4/talking_head_testing/wavs_16000'

# vid_src = 'align_crop'
# vid_target = 'align_crop_direct_audio'
# wav_rt = '/data4/talking_head_testing/wavs_extract_from_newmead'

# vid_src = 'align_crop'
# vid_target = 'align_crop_direct_audio'
# wav_rt = '/data4/talking_head_testing/wavs_extract_from_newmead'

vid_src = 'no_crop'
vid_target = 'no_crop_direct_audio'
wav_rt = '/data4/talking_head_testing/wavs_extract_from_newmead'

if __name__ == '__main__':
    args = get_args()
    vid_pths = glob.glob('/data4/talking_head_testing/25fps_video/{}/{}/*.mp4'.format(vid_src,args.name))
    out_vid_rt = '/data4/talking_head_testing/25fps_video/{}/{}'.format(vid_target,args.name)
    os.makedirs(out_vid_rt,exist_ok=True)

    for vid_pth in tqdm.tqdm(vid_pths):
        fname = vid_pth.split('/')[-1].split('.')[0]
        wname = fname
        # wname = fname.replace('_','-')
        wav_pth = '{}/{}.wav'.format(wav_rt,wname)
        out_pth = '{}/{}.mp4'.format(out_vid_rt,fname)
        
        fuse_cmds = 'ffmpeg -loglevel quiet -y -i {} -i {} -vcodec copy {}'.format(vid_pth,wav_pth,out_pth)
        os.system(fuse_cmds)
        # assert(0)

# python fuse_audio_on_croped_videos.py --name gt
# python fuse_audio_on_croped_videos.py --name EAMM
# python fuse_audio_on_croped_videos.py --name EAMM_pcavs
# python fuse_audio_on_croped_videos.py --name test-523
# python fuse_audio_on_croped_videos.py --name mit
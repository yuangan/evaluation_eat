import os
import glob

wav_pths = glob.glob( '/data3/shared/MEAD/*/wav/*/level_*/*.wav')
rt_pth = '/data4/talking_head_testing/wavs_16000'
for wav_pth in wav_pths:
    _, _ , _, _ , pid, _, emo, lev , vid =  wav_pth.split('/')
    vid = vid.split('.')[0]
    wname = '{}-{}-{}-{}'.format(pid,emo[:3],lev[-1],vid)
    out_pth = '{}/{}.wav'.format(rt_pth,wname)
    cmd = 'ffmpeg -i {} -ar 16000 {}'.format(wav_pth,out_pth)
    os.system(cmd)


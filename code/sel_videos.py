import os


vid_rt = '/data4/talking_head_testing/25fps_video/no_crop'

methods = [ 'atvg_0913' , 'audio2head_0913' , 'EAMM' , 'gt' , 'maketitalk_0913' , 'pcavs' , 'wav2lip_0913' , 'ours_new_gtpose' ]

# vnames = ['M003_neu_1_013','M003_fea_3_009','W015_con_3_023','W009_sad_3_029','W009_ang_3_022','W015_hap_3_014','M030_sur_3_023','M030_dis_3_019']
# vnames = ['M003_hap_3_026','M030_ang_3_014','W015_con_3_004']

vnames = ['M003_con_3_020','W015_hap_3_010']

sav_rt = '/data4/talking_head_testing/sel_videos2'

for method in methods:
    savp = '{}/{}'.format(sav_rt,method)
    os.makedirs(savp,exist_ok=True)
    for vname in vnames:
        cmd = 'cp {}/{}/{}.mp4 {}/{}.mp4'.format(vid_rt,method,vname,savp,vname)
        os.system(cmd)
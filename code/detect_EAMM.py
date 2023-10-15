import glob
import os


# id_dirs = glob.glob('/data4/makeittalk_other_res/res/*')

# mi = 1e9+7
# vid_num = 0
# for id_d in id_dirs:
#     vids = glob.glob( '{}/*'.format(id_d) )
#     # print(len(vids))
#     for vid in vids:
#         vid_num += 1
#         frms = glob.glob( '{}/*'.format(vid) )
#         mi = min( mi , len(frms) )

# print(mi,vid_num)

s2l = { 'ang': 'angry' , 'con': 'contempt' , 'dis': 'disgusted' , 'fea': 'fear' , 'hap': 'happy' , 'neu': 'neutral' , 'sad': 'sad' , 'sur': 'surprised' }

vids = glob.glob('/data4/new_mead/video_test/*.mp4')

for vpth in vids:
    pid,emo,lev,vid = vpth.split('/')[-1].split('.')[0].split('_')
    
    eamm_pth = '/data4/makeittalk_other_res/res/{}/{}_level_{}_{}'.format(pid,s2l[emo],lev,vid)
    if not os.path.exists(eamm_pth):
        print( pid,emo,lev,vid )
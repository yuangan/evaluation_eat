import argparse
from cmath import inf
from turtle import distance
import cv2
import dlib
import numpy as np
from imutils import face_utils
import math
import os
import glob
import imageio

INF = 1e9+7

emo_name = ['angry' , 'happy' , 'fear' , 'neutral'  ,'sad' , 'surprised' , 'disgusted' , 'contempt']
emo_pre_to_name = { 'ang': 'angry' , 'hap': 'happy' , 'fea': 'fear' , 'neu': 'neutral'  , 'sad' : 'sad' , 'sur': 'surprised' , 'dis': 'disgusted' , 'con': 'contempt' }

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
(mouth_Start, mouth_End) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
# print(face_utils.FACIAL_LANDMARKS_IDXS.keys()) # odict_keys(['mouth', 'inner_mouth', 'right_eyebrow', 'left_eyebrow', 'right_eye', 'left_eye', 'nose', 'jaw'])

def get_pth_gt2( pid , emo , lev , vid , gtname = 'gt'  ): 
    return '../talking_head_testing/25fps_video/align_crop/{}/{}_{}_{}_{}.mp4'.format(gtname, pid , emo , lev , vid)


def visulize(lmk1,lmk2,fid=0):
    white_img = np.ones( (256,256,3) ) * 255
    # print(lmk1,lmk2)
    for lmk in lmk1: cv2.circle(white_img,(int(lmk[0]),int(lmk[1])),1,(255,0,0))
    for lmk in lmk2: cv2.circle(white_img,(int(lmk[0]),int(lmk[1])),1,(0,0,255))

    cv2.imwrite('./vis/{}.jpg'.format(fid),white_img)

# get mouth lmk for single frame
def get_lmk(img):
    # attention , it is not the BGR type for the img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    # loop over the face detections
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        mouth = shape[mouth_Start:mouth_End]
        return mouth, shape
    return None,None

# calibrate the lmk to the mean location
def calibrate_lmk(lmk,adjust_x = -1 , adjust_y = -1):
    if lmk is None : return None , None , None
    lmk = lmk.astype(np.float64)
    mean_x = np.mean(lmk[:,0]) 
    mean_y = np.mean(lmk[:,1])
    lmk[:,0] = lmk[:,0] - mean_x
    lmk[:,1] = lmk[:,1] - mean_y
    xlen = lmk[:,0].max() - lmk[:,0].min()
    ylen = lmk[:,1].max() - lmk[:,1].min()
    if adjust_x != -1 : lmk[:,0] = lmk[:,0] / xlen * adjust_x
    if adjust_y != -1 : lmk[:,1] = lmk[:,1] / ylen * adjust_y
    xlen = lmk[:,0].max() - lmk[:,0].min()
    ylen = lmk[:,1].max() - lmk[:,1].min()
    return lmk , xlen  , ylen

def euc_dis(p1,p2):
    dis = math.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )
    # print(p1,p2,dis)
    return dis


def ld(lmk1,lmk2):
    lmk1 , x_len1 , y_len1 = calibrate_lmk( lmk1 ) # 20 * 2 ndarray
    lmk2 , x_len2 , y_len2  = calibrate_lmk( lmk2  )
    # print( x_len1 / x_len2  )
    # print( y_len1 / y_len2  )
    if lmk1 is None or lmk2 is None: return -1
    
    '''adjust for atvg'''
    # v1_mouth_lmk[:,0] = v1_mouth_lmk[:,0] * 1.125
    # v1_mouth_lmk[:,1] = v1_mouth_lmk[:,1] * 1.125

    # visulize(v1_mouth_lmk+128,v2_mouth_lmk+128,fid)
    dis_sum = 0
    for i , p1 in enumerate(lmk1):
        dis_sum += euc_dis(p1,lmk2[i])
    return dis_sum

def lvd(dert_lmk1,dert_lmk2):

    if dert_lmk1 is None or dert_lmk2 is None: return -1

    dis_sum = 0
    for i , p1 in enumerate(dert_lmk1):
        dis_sum += euc_dis(p1,dert_lmk2[i])
    return dis_sum


def load_frame_lis(vid_pth):
    img_lis = []
    if os.path.isdir(vid_pth) : 
        frame_lis = os.listdir(vid_pth)
        try :
            frame_lis.sort(key = lambda x : int( x.split('.')[0] ))
        except : return None
        for frame_name in frame_lis:
            img_lis.append( cv2.cvtColor( cv2.imread( '{}/{}'.format(vid_pth,frame_name) ) , cv2.COLOR_RGB2BGR ) )
    else :
        fake_reader = imageio.get_reader( vid_pth )
        for im in fake_reader: img_lis.append( im )
    return img_lis

# compute lmd for two videos
# TODO : how to process img size problem ?
# TODO : some videos may be list of imgs.
# attention. set v2 is the gt, which offers the std scale!
def get_lmd(v1_pth,v2_pth,bool_only_96):
    # v1_reader = cv2.VideoCapture(v1_pth)
    # fv1 = int(v1_reader.get(7))
    # # print(v2_pth)
    # if not v2_pth.endswith('.mp4'):
    #     # print(v2_pth)
    #     fv2 = len( glob.glob( v2_pth + '/' + '*.jpg'  ) )
    #     # print(fv2)
    # else :
    #     v2_reader = cv2.VideoCapture(v2_pth)
    #     fv2 = int(v2_reader.get(7))
    # print('frame_v1:',v1_reader.get(7))
    # print('frame_v2',v2_reader.get(7))
    # print(fv1,fv2)
    # atten!!!! TODO different processing ways for ours and others!
    
    v1_lis = load_frame_lis(v1_pth)
    v2_lis = load_frame_lis(v2_pth)
    if v1_lis is None or v2_lis is None : return -1,-1,-1,-1
    fv1 , fv2 = len(v1_lis) , len(v2_lis)
    
    # bool_only_96 = True # True for ours & fsv2v , False for mit & atvg
    if bool_only_96 :
        fv2 = min(fv2 , 96) 
        v2_lis = v2_lis[:fv2]


    frame_len = int(min( fv1,fv2 ))
    if frame_len == 0 : return -1 , -1 , -1 , -1

    
    # print(frame_len)
    fv1_lis = np.linspace(0,fv1,frame_len,False).astype(np.int32).tolist()
    fv2_lis = np.linspace(0,fv2,frame_len,False).astype(np.int32).tolist()
    # print(fv1,fv1_lis)
    # print(fv2,fv2_lis)

    mouth_dis_sum , face_dis_sum , mouth_lvd_sum , face_lvd_sum = 0 , 0 , 0 ,0
    # mouth_lmk_num , face_lmk_num = -1 , -1
    valid_num = 0
    last_mouth_lmk1 , last_mouth_lmk2  = None , None
    last_face_lmk1 , last_face_lmk2 = None , None
    for fid in range(0,frame_len):
        v1_frame, v2_frame = v1_lis[fv1_lis[fid]], v2_lis[fv2_lis[fid]]
#       v1_reader.set(cv2.CAP_PROP_POS_FRAMES, fv1_lis[fid])
#       v1_ret , v1_frame = v1_reader.read()
#       if not v2_pth.endswith('.mp4'):
#           im_pth = v2_pth + '/' + ( '%04d'%(fid+1) ) + '.jpg'
#           v2_frame = cv2.imread( im_pth )
#       else :
#           v2_reader.set(cv2.CAP_PROP_POS_FRAMES, fv2_lis[fid])
#           v2_ret , v2_frame = v2_reader.read()

        mouth_lmk1 , face_lmk1 = get_lmk(v1_frame)
        if mouth_lmk1 is None or face_lmk1 is None: continue
        mouth_lmk2 , face_lmk2 = get_lmk(v2_frame)
        if mouth_lmk2 is None or face_lmk2 is None: continue

        mouth_dis = ld(mouth_lmk1,mouth_lmk2)
        if mouth_dis == -1 : continue
        face_dis = ld(face_lmk1,face_lmk2)
        if face_dis == -1 : continue

        mouth_dis_sum += mouth_dis
        face_dis_sum += face_dis

        if last_mouth_lmk1 is not None: dert_mouth_lmk1 = mouth_lmk1 - last_mouth_lmk1
        if last_mouth_lmk2 is not None: dert_mouth_lmk2 = mouth_lmk2 - last_mouth_lmk2
        if last_face_lmk1 is not None: dert_face_lmk1 = face_lmk1 - last_face_lmk1
        if last_face_lmk2 is not None: dert_face_lmk2 = face_lmk2 - last_face_lmk2
        
        if last_mouth_lmk1 is not None and last_mouth_lmk2 is not None:
            lvd_mouth = lvd(dert_mouth_lmk1,dert_mouth_lmk2)
            mouth_lvd_sum += lvd_mouth
        if last_face_lmk1 is not None and last_face_lmk2 is not None:
            lvd_face = lvd(dert_face_lmk1,dert_face_lmk2)
            face_lvd_sum += lvd_mouth

        last_mouth_lmk1 = mouth_lmk1
        last_mouth_lmk2 = mouth_lmk2
        last_face_lmk1 = face_lmk1
        last_face_lmk2 = face_lmk2

        valid_num += 1


    if valid_num == 0 : return  -1 , -1 , -1 , -1   # unvalid video

    return mouth_dis_sum / ( valid_num * 20 ) , mouth_lvd_sum / ( valid_num * 20 )  , face_dis_sum / ( valid_num * 68 ) , face_lvd_sum / ( valid_num * 68 )


# if __name__ == '__main__':
#     lmd = get_lmd('/data3/shared/MEAD/M031/video/front/surprised/level_3/003.mp4','/data3/shared/MEAD/M031/video/front/surprised/level_3/003.mp4')
#     print('lmd:',lmd)

def pro_vid_1(vp,name_mode,bool_only96,gtname='gt'):
    # print(vp.split('_'))
    vname = os.path.splitext(os.path.split(vp)[-1])[0]
    print(vname)
    f = vp
    if name_mode == 0 : 
        # EAMM (for dirs)
        pid , emo_lev_vid = f.split('/')[-2] , f.split('/')[-1]
        emo, _ , lev , vid = emo_lev_vid.split('_')
        emo = emo[:3]
    elif name_mode == 1 :  
        # makeittalk
        pid , emo , lev , vid , _  =  os.path.splitext( os.path.split(f)[1] )[0].split('_')
    elif name_mode == 2 : 
        # ATVG
        pid , emo , lev , vid  =  os.path.splitext( os.path.split(f)[1] )[0].split('_')
    elif name_mode == 3 : 
        pid , emo , lev , vid = f.split('-')
    elif name_mode == 4 : _ , pid , emo , lev , vid  =  os.path.splitext( os.path.split(f)[1] )[0].split('_') 
    elif name_mode == 5:
        # EAMM (for processed vid)
        pid, emo, _ , lev , vid = os.path.splitext( os.path.split(f)[1] )[0].split('_') 
        emo = emo[:3]
    elif name_mode == 6:
        pid, emo, lev , vid = os.path.split(f)[1].split('.')[0].split('_')
    
    gt_vp = get_pth_gt2( pid , emo , lev , vid , gtname = gtname )
    
    # pid = pid.split('/')[-1]
    # gt_vp = '/data3/shared/MEAD/{}/video/front/{}/level_{}/{}.mp4'.format(pid,emo_pre_to_name[emo],lev,vid)
    # gt_vp = '/data4/MEAD_cropped/videos/{}_{}_{}_{}.mp4'.format(pid,emo,lev,vid)
    # print(gt_vp)
    # if lev == 1 : return -1
    # gt_vp = '/data4/MEAD_preprocess_for_ATVGnet/for_train_vid2/{}-{}-lev{}-{}'.format(pid,emo,lev,vid)
    return get_lmd(vp,gt_vp,bool_only96)

def get_parse():
    args = argparse.ArgumentParser('psnr_ssim')
    args.add_argument('--save_name',type=str)
    args.add_argument('--fake_pth',type=str)
    args.add_argument('--name_mode',type=int,default=6)
    args.add_argument('--bool_only96',action='store_true')
    args.add_argument('--gtname',type=str,default='evp_gt')
    return args

if __name__ == '__main__':
    '''ours'''
    # vlis = glob.glob('/data4/lmd-test/lmd-videos/output/test_lmd_videos/*.mp4')
    '''makeittalk'''
    # vlis = glob.glob('/data4/MEAD_preprocess_for_MakeItTalk/_new_out_vids/*.mp4')
    # vlis = glob.glob('/data4/MEAD_preprocess_for_MakeItTalk/_new_out_vids/M009_fea_3_001_gen.mp4')
    '''makeittalk - amend'''
    # vlis = glob.glob('/data4/MEAD_res/mead-amend-vids/*.mp4')
    '''atvg net'''
    # vlis = glob.glob('/data4/MEAD_preprocess_for_ATVGnet/_new_out_vids/*.mp4')
    '''mead gt'''
    # vlis = glob.glob('/data4/MEAD_cropped/videos/train/*.mp4')
    '''ablation_kp'''
    # vlis = glob.glob('/data4/lmd-test/ablation_kp/output/test_lmd_videos/*.mp4')
    '''ablation_full'''
    # vlis = glob.glob('/data4/lmd-test/ablation_full/output/test_lmd_videos/*.mp4')
    '''ablation_emoclassify'''
    # vlis = glob.glob('/data4/lmd-test/ablation_emoclassify/output/test_lmd_videos/*.mp4')
    '''ours-new'''
    # vlis = glob.glob('/data4/EADG_res/ours_mead_test_389/test_lmd_videos/*.mp4')
    '''fsv2v'''
    # vlis = glob.glob('/data4/fsv2v_gen/output/gt_lmd_videos/*.mp4')
    '''ours-new-434'''
    # vlis = glob.glob('/data4/EADG_res/ours_mead_test_434/test_lmd_videos_434/*.mp4')
    '''ours-new-494'''
    # vlis = glob.glob('/data4/EADG_res/test_lmd_videos_494/test_lmd_videos_494/*.mp4')
    '''ours-wopca'''
    # vlis = glob.glob('/data4/EADG_res/wopca_retrain/test_lmd_videos_wopca_retrain/*.mp4')
    '''ours-withvox'''
    # vlis = glob.glob('/data4/EADG_res/ablation_e_withvox/test_lmd_videos_ablation_e/*.mp4')
    '''test_lmd_fixpose_videos_334'''
    # vlis = glob.glob('/data4/EADG_res/test_lmd_fixpose_videos_334/test_lmd_fixpose_videos_334/*.mp4')
    '''test_lmd_fixpose_videos_464'''
    # vlis = glob.glob('/data4/EADG_res/test_lmd_fixpose_videos_464/test_lmd_fixpose_videos_464/*.mp4')
    
    args = get_parse().parse_args()
    vlis = glob.glob(args.fake_pth)
    print(len(vlis))
    # breakpoint()
    name_mode = args.name_mode
    save_name = args.save_name
    bool_only96 = args.bool_only96
    # print(bool_only96)
    # vlis = vlis[0:4]
    mouth_lmd_sum , mouth_valid_vid = 0 , 0
    face_lmd_sum  , face_valid_vid = 0 , 0
    face_lvd_sum  , face_lvd_valid_vid = 0 , 0
    mouth_lvd_sum  , mouth_lvd_valid_vid = 0 , 0
    with open('result/{}.txt'.format(save_name),'w') as f:
        for vp in vlis:
            mouth_lmd , mouth_lvd , face_lmd , face_lvd = pro_vid_1(vp,name_mode,bool_only96,gtname=args.gtname)
            if mouth_lmd != -1 : 
                mouth_lmd_sum += mouth_lmd 
                mouth_valid_vid += 1
            if face_lmd != -1 : 
                face_lmd_sum += face_lmd 
                face_valid_vid += 1
            if mouth_lvd != -1 : 
                mouth_lvd_sum += mouth_lvd 
                mouth_lvd_valid_vid += 1
            if face_lvd != -1 : 
                face_lvd_sum += face_lvd 
                face_lvd_valid_vid += 1
            # print(vp , tmp_lmd)
            f.writelines( '{} : {} , {} , {} , {} \n '.format(vp,mouth_lmd,face_lmd,mouth_lvd,face_lvd) )
            f.flush()
        # print( 'lmd:' , sum/len(vlis) )
        f.writelines( 'vid num:' + str( len(vlis) ) + '\n' )
        f.writelines( 'mouth valid vid num:' + str( mouth_valid_vid ) + '\n' )
        f.writelines( 'mouth lmd:' + str( mouth_lmd_sum/mouth_valid_vid ) + '\n' )
        f.writelines( 'face valid vid num:' + str( face_valid_vid ) + '\n' )
        f.writelines( 'face lmd:' + str( face_lmd_sum/face_valid_vid ) + '\n' )
        f.writelines( 'mouth valid lvd vid num:' + str( mouth_lvd_valid_vid ) + '\n' )
        f.writelines( 'mouth lvd:' + str( mouth_lvd_sum/mouth_lvd_valid_vid ) + '\n' )
        f.writelines( 'face valid lvd vid num:' + str( face_lvd_valid_vid ) + '\n' )
        f.writelines( 'face lvd:' + str( face_lvd_sum/face_lvd_valid_vid ) + '\n' )

# /data4/MEAD_cropped/videos/

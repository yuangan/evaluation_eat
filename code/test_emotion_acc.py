from distutils.util import execute
import os
import glob

from torch._C import TensorType, _tracer_warn_use_python
import data.face_alignment_code._acc_test_frame2face as frame2face
import data.face_alignment_code._acc_test_video2frame as video2frame
import subprocess

import argparse

def get_args():
    parser = argparse.ArgumentParser('test emotion')
    parser.add_argument('--in_vid_path',type=str)
    parser.add_argument('--save_name',type=str)
    parser.add_argument('--emo_range_l',type=int,default=5)
    parser.add_argument('--emo_range_r',type=int,default=8)
    parser.add_argument('--gpu_id',type=int,default=0)
    parser.add_argument('--start_state',type=int,default=1)
    parser.add_argument('--model',type=int,default=0)
    parser.add_argument('--mid_file_dir',type=str,default='emoemo_general_testing')

    return parser.parse_args()

# python _acc_test_pipeline.py --in_vid_path /data4/talking_head_testing/temp_res/EADG/vt2mel25_vox_head_507 --save_name EADG_507 --emo_range_l 10 --emo_range_r 13

# for training
# 1. excute_test = False 2. set output path 
# python _acc_test_pipeline.py --in_vid_path /data3/vox/vox/mead/videos_evp_25 --save_name preprocess_for_training --emo_range_l 5 --emo_range_r 8

def main():
    # arguments

    # in_vid_path    = '/data4/test_data_for_emotion-fan/test29_video/'
    # mid_frame_path = '/data4/on_generation_fake_videos/from_neutral-noft/'
    # out_face_path  = '/data4/test_data_for_emotion-fan/test31_face/'
    # train_list_filename = '/data4/test_data_for_emotion-fan/test-face-test31_face.txt' #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！
    # # 29 : change audio - test_oneshot_gen_align_obama 
    # 30 : from_neutral 
    # 16 : facev2v - ft 
    # 

    # in_vid_path    = '/data4/emotion-fan-single-test/video/'
    # mid_frame_path = '/data4/emotion-fan-single-test/frame/'
    # out_face_path  = '/data4/emotion-fan-single-test/face/'
    # train_list_filename = '/data4/emotion-fan-single-test/train_list.txt' #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！

    # in_vid_path    = '/data4/EADG_res/allwav_15/test_eadg_largesize_494_allwav_15/'
    # mid_frame_path = '/data4/test_data_for_emotion/ours-allwav_15/frame/'
    # out_face_path  = '/data4/test_data_for_emotion/ours-allwav_15/face/'
    # train_list_filename = '/data4/test_data_for_emotion/ours-allwav_15/train_list_4.txt' #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！

    
    # in_vid_path    = '/data4/MEAD_res/oneshot-new-1-vids/'
    # mid_frame_path = '/data4/test_data_for_emotion/mit--oneshot-new-1-vids/frame/'
    # out_face_path  = '/data4/test_data_for_emotion/mit--oneshot-new-1-vids/face/'
    # train_list_filename = '/data4/test_data_for_emotion/mit--oneshot-new-1-vids/train_list.txt' #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！
    
    # atvg
    # in_vid_path    = '/data4/ATVG_res/atvg-new-vids-amend/'
    # mid_frame_path = '/data4/test_data_for_emotion/atvg-new-vids-amend-all/frame/'
    # out_face_path  = '/data4/test_data_for_emotion/atvg-new-vids-amend-all/face/'
    # train_list_filename = '/data4/test_data_for_emotion/atvg-new-vids-amend-all/train_list4.txt' #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！
    
    # fsv2v
    # in_vid_path    = '/data4/fsv2v_gen/test_eadg_largesize_facev2v_wonorm_allwav_15/test_eadg_largesize_facev2v_wonorm_allwav_15/'
    # mid_frame_path = '/data4/test_data_for_emotion/fsv2v-allwav15-all/frame/'
    # out_face_path  = '/data4/test_data_for_emotion/fsv2v-allwav15-all/face/'
    # train_list_filename = '/data4/test_data_for_emotion/fsv2v-allwav15-all/train_list4.txt' #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！
    
    # makeittalk
    # in_vid_path    = '/data4/MEAD_res/oneshot-new-1-vids/'
    # mid_frame_path = '/data4/test_data_for_emotion/mit-allwav15-all/frame/'
    # out_face_path  = '/data4/test_data_for_emotion/mit-allwav15-all/face/'
    # train_list_filename = '/data4/test_data_for_emotion/mit-allwav15-all/train_list3.txt' #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！
    
    # fsv2v norm
    # in_vid_path    = '/data4/fsv2v_gen/face-v2v-ft-norm-allwav_15/'
    # mid_frame_path = '/data4/test_data_for_emotion/fsv2v-norm-allwav15-all/frame/'
    # out_face_path  = '/data4/test_data_for_emotion/fsv2v-norm-allwav15-all/face/'
    # train_list_filename = '/data4/test_data_for_emotion/fsv2v-norm-allwav15-all/train_list3.txt' #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！
    
    # TODO waiting to process~ 
    # in_vid_path    = '/data4/EADG_res/ablation_494_allwav_15/ablation_largesize_494_allwav_15/'
    # mid_frame_path = '/data4/test_data_for_emotion/ours-ablation_494_allwav_15/frame/'
    # out_face_path  = '/data4/test_data_for_emotion/ours-ablation_494_allwav_15/face/'
    # train_list_filename = '/data4/test_data_for_emotion/ours-ablation_494_allwav_15/train_list.txt' #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！
    
    # crema-d fsv2v-norm
    # in_vid_path    = '/data4/fsv2v_gen/face-v2v-ft-norm-allwav_15_cremad/'
    # mid_frame_path = '/data4/test_data_for_emotion/cremad-fv2v-norm/frame/'
    # out_face_path  = '/data4/test_data_for_emotion/cremad-fv2v-norm/face/'
    # train_list_filename = '/data4/test_data_for_emotion/cremad-fv2v-norm/train_list.txt' #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！
    
    # crema-d fsv2v
    # in_vid_path    = '/data4/fsv2v_gen/cremad_eat_largesize_fv2v-ft/cremad_test_eadg_largesize_facev2v_wonorm_allwav_15/'
    # mid_frame_path = '/data4/test_data_for_emotion/cremad-fv2v/frame/'
    # out_face_path  = '/data4/test_data_for_emotion/cremad-fv2v/face/'
    # train_list_filename = '/data4/test_data_for_emotion/cremad-fv2v/train_list.txt' #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！
    
    # in_vid_path    = '/data4/fsv2v_gen/face-v2v-ft-allwav_15_cremad/cremad_test_eadg_largesize_facev2v_wonorm_allwav_15/'
    # mid_frame_path = '/data4/test_data_for_emotion/cremad-fv2v-amend/frame/'
    # out_face_path  = '/data4/test_data_for_emotion/cremad-fv2v-amend/face/'
    # train_list_filename = '/data4/test_data_for_emotion/cremad-fv2v-amend/train_list.txt' #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！

    # crema-d mit-sample
    # sample_id = 9
    # in_vid_path    = '/data4/CREMAD_res/sample_res/{}_vids'.format(sample_id)
    # mid_frame_path = '/data4/test_data_for_emotion/cremad-mit-sample-{}/frame/'.format(sample_id)
    # out_face_path  = '/data4/test_data_for_emotion/cremad-mit-sample-{}/face/'.format(sample_id)
    # train_list_filename = '/data4/test_data_for_emotion/cremad-mit-sample-{}/train_list.txt'.format(sample_id) #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！

    # crema-d mit
    # in_vid_path    = '/data4/CREMAD_res/all/0_vids/'
    # mid_frame_path = '/data4/test_data_for_emotion/cremad-mit/frame/'
    # out_face_path  = '/data4/test_data_for_emotion/cremad-mit/face/'
    # train_list_filename = '/data4/test_data_for_emotion/cremad-mit/train_list.txt' #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！

    # crema-d atvg
    # in_vid_path    = '/data4/CREMAD-ATVGres/out_vids/'
    # mid_frame_path = '/data4/test_data_for_emotion/cremad-atvg/frame/'
    # out_face_path  = '/data4/test_data_for_emotion/cremad-atvg/face/'
    # train_list_filename = '/data4/test_data_for_emotion/cremad-atvg/train_list.txt' #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！

    # crema-d ours
    # in_vid_path    = '/data4/EADG_res/cremad_allwav_15/test_eadg_largesize_299_allwav_15_cremad'
    # mid_frame_path = '/data4/test_data_for_emotion/cremad-ours/frame/'
    # out_face_path  = '/data4/test_data_for_emotion/cremad-ours/face/'
    # train_list_filename = '/data4/test_data_for_emotion/cremad-ours/train_list.txt' #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！


    # crema-d gt
    # in_vid_path    = ''
    # mid_frame_path = '/data3/shared/CREMA-D-crop/test/'
    # out_face_path  = '/data4/test_data_for_emotion/cremad-test-gt/face/'
    # train_list_filename = '/data4/test_data_for_emotion/cremad-test-gt/train_list.txt' #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！

    # ours-dead
    # in_vid_path    = '/data4/EADG_res/test_eadg_largesize_494_allwav_15_cremad/test_eadg_largesize_494_allwav_15_cremad/'
    # mid_frame_path = '/data3/test_data_for_emotion/ours-dead-494/frame/'
    # out_face_path  = '/data4/test_data_for_emotion/ours-dead-494/face/'
    # train_list_filename = '/data4/test_data_for_emotion/ours-dead-494/train_list.txt' #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！

    # ours-dead-299-2300
    # in_vid_path    = '/data4/EADG_res/test_eadg_largesize_299_allwav_15_cremad_2300/test_eadg_largesize_299_allwav_15_cremad_2300/'
    # mid_frame_path = '/data3/test_data_for_emotion/ours-dead-299-2300/frame/'
    # out_face_path  = '/data4/test_data_for_emotion/ours-dead-299-2300/face/'
    # train_list_filename = '/data4/test_data_for_emotion/ours-dead-299-2300/train_list.txt' #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！

    # ours-dead-494-2300
    # in_vid_path    = '/data4/EADG_res/test_eadg_largesize_494_allwav_15_cremad_2300/test_eadg_largesize_494_allwav_15_cremad_2300/'
    # mid_frame_path = '/data3/test_data_for_emotion/ours-dead-494-2300/frame/'
    # out_face_path  = '/data4/test_data_for_emotion/ours-dead-494-2300/face/'
    # train_list_filename = '/data4/test_data_for_emotion/ours-dead-494-2300/train_list.txt' #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！

    # ours - MEAD
    # in_vid_path    = '/data4/EADG_res/test_lmd_videos_494/test_lmd_videos_494/'
    # mid_frame_path = '/data3/test_data_for_emotion/ours-mead-test_lmd_videos_494/frame/'
    # out_face_path  = '/data4/test_data_for_emotion/ours-mead-test_lmd_videos_494/face/'
    # train_list_filename = '/data4/test_data_for_emotion/ours-mead-test_lmd_videos_494/train_list.txt' #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！

    # mit - MEAD
    # in_vid_path    = '/data4/MEAD_preprocess_for_MakeItTalk/_new_out_vids/'
    # mid_frame_path = '/data4/test_data_for_emotion/mit-mead/frame/'
    # out_face_path  = '/data4/test_data_for_emotion/mit-mead/face/'
    # train_list_filename = '/data4/test_data_for_emotion/mit-mead/train_list.txt' #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！

    # atvg - MEAD
    # in_vid_path    = '/data4/MEAD_preprocess_for_ATVGnet/_new_out_vids/'
    # mid_frame_path = '/data4/test_data_for_emotion/atvg-mead-neu/frame/'
    # out_face_path  = '/data4/test_data_for_emotion/atvg-mead-neu/face/'
    # train_list_filename = '/data4/test_data_for_emotion/atvg-mead-neu/train_list.txt' #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！
    
    # fsv2v - MEAD
    # in_vid_path    = '/data4/fsv2v_gen/fv2v-ft_mead_test/gt_lmd_videos/'
    # mid_frame_path = '/data4/test_data_for_emotion/fsv2v-mead/frame/'
    # out_face_path  = '/data4/test_data_for_emotion/fsv2v-mead/face/'
    # train_list_filename = '/data4/test_data_for_emotion/fsv2v-mead/train_list.txt' #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！
    
    args = get_args()
    in_vid_path    = args.in_vid_path + '/'
    mid_frame_path = '../test_data_for_emotion/{}/frames/{}/'.format(args.mid_file_dir,args.save_name)
    out_face_path  = '../test_data_for_emotion/{}/faces/{}/'.format(args.mid_file_dir,args.save_name)
    train_list_filename = '../test_data_for_emotion/{}/train_list/{}.txt'.format(args.mid_file_dir,args.save_name) #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！
    
    # only for training
    # mid_frame_path = '/data4/test_data_for_emotion/for_training/frames/'
    # out_face_path  = '/data4/test_data_for_emotion/for_training/faces/'
    # train_list_filename = '/data4/test_data_for_emotion/for_training/train_list.txt' #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！
    

    video2frame_threads_num = 5
    frame2face_threads_num = 5

    gpu_id = args.gpu_id

    # model_path = './model_lev123_disid/self_relation-attention_4_65.4348'
    model_path = 'model_training_name/self_relation-attention_61_84.3655'
    #if args.model == 1: model_path = 'model_new_mead_1008_disid/self_relation-attention_1_82.5381'
    #if args.model == 2: model_path = 'model_training_name/self_relation-attention_61_84.3655'

    print(model_path)
    # ls , rs = 15 , 18 # 表情绪的名字区间 crema fsv2v
    # ls , rs = 6 , 9 # 表情绪的名字区间 crema mit sample , mit , ours
    # ls , rs = 5 , 8 # 表情绪的名字区间 crema mit sample , mit , ours
    # ls , rs = 9 , 12 # 表情绪的名字区间 crema gt
    # ls , rs = 17 , 20 # 表情绪的名字区间 crema mit sample , mit
    ls, rs = args.emo_range_l, args.emo_range_r

    start_state = args.start_state # 0 要从.98服务器copy来vid 1 现有vid，要frame 2 现有frame，要face  3 现有face，要train_list 4 现有train_list,直接测试
    execute_test = True # False for 5000  ; True for 1000

    # video to frame
    if start_state == 1:
        vlist = glob.glob(in_vid_path+'/'+'*.mp4')
        print(len(vlist))
        threads = []
        for vpth in vlist:
            out_path = mid_frame_path + '/' + os.path.splitext( os.path.split(vpth)[1] )[0]
            os.makedirs(out_path,exist_ok=True)
            threads.append( video2frame.threadFun(  video2frame.video2frame , ( vpth , out_path )  ) )
        video2frame.run_threads(threads,video2frame_threads_num) 
        start_state += 1

    # frame to face
    if start_state == 2:
        frame2face.main( mid_frame_path , out_face_path , frame2face_threads_num )
        start_state += 1

    # get train list
    if start_state == 3:
        label_map = { 'sad' : 'Sad' , 'con' : 'Contempt' , 'fea' : 'Fear' , 'hap' : 'Happy' , 'dis' : 'Disgust' , 'sur' : 'Surprised' , 'ang' : 'Angry' , 'neu' : 'Neutral'  }
        vid_dir_list = glob.glob(out_face_path+'/*') 
        print(out_face_path)
        with open(train_list_filename,'w') as f: 
            for dname in vid_dir_list: 
                _ , dname = os.path.split(dname) 
                print(dname)
                # s = dname.split('_')[6][0:3]
                # if s == 'boy' : s = dname.split('_')[2]
                # if s == 'boy2' : s = dname.split('_')[2]
                s = dname[ls:rs].lower()
                # s = 'neu'
                typ = label_map[ s ] 
                # print(dname , typ) 
                f.write(dname+' '+typ+'\n') 
        print('root path:',out_face_path) 
        print('list txt:',train_list_filename)         
        start_state += 1    

    # do test , start_state = 4
    if start_state == 4 and execute_test:
        cmd = 'CUDA_VISIBLE_DEVICES='+str(gpu_id)+' python _acc_test.py -e --lr 4e-7 --save_name '+ args.save_name +' --name acc_testdata -m '+model_path+' --reval '+out_face_path+' --leval '+train_list_filename
        print('run command:',cmd)
        subprocess.call(cmd,shell=True)

if __name__ == '__main__':
    main()



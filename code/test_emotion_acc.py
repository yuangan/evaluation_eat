from distutils.util import execute
import os
import glob

from torch._C import TensorType, _tracer_warn_use_python
import face_alignment_code._acc_test_frame2face as frame2face
import face_alignment_code._acc_test_video2frame as video2frame
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

def main():
    args = get_args()
    in_vid_path    = args.in_vid_path + '/'
    mid_frame_path = './test_data_for_emotionacc/{}/frames/{}/'.format(args.mid_file_dir,args.save_name)
    out_face_path  = './test_data_for_emotionacc/{}/faces/{}/'.format(args.mid_file_dir,args.save_name)
    train_list_filename = './test_data_for_emotionacc/{}/train_list/{}.txt'.format(args.mid_file_dir,args.save_name) #  注意，若start_state<=3，程序运行后，该目录原有内容会被清空！
    
    video2frame_threads_num = 5
    frame2face_threads_num = 5

    gpu_id = args.gpu_id

    model_path = 'model_training_name/self_relation-attention_61_84.3655'

    print(model_path)
    # ls , rs = 15 , 18 # 表情绪的名字区间 crema fsv2v
    # ls , rs = 6 , 9 # 表情绪的名字区间 crema mit sample , mit , ours
    # ls , rs = 5 , 8 # 表情绪的名字区间 crema mit sample , mit , ours
    # ls , rs = 9 , 12 # 表情绪的名字区间 crema gt
    # ls , rs = 17 , 20 # 表情绪的名字区间 crema mit sample , mit
    ls, rs = args.emo_range_l, args.emo_range_r

    start_state = args.start_state
    execute_test = True

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
                s = dname[ls:rs].lower()
                typ = label_map[ s ] 
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



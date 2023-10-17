#name='qvt_img_pca_sync_prompt2_05_480_new_neutral'
#name='EAMM'
name=$1
device=$2

cd './code'
#----------------------------------------------##
# preprocess takes 7 min
python preprocess.py --save_name ${name}_Eval --fake_pth "../result/${name}/*.mp4" --name_mode 4  --ours_filter_100 --need_align_crop
#----------------------------------------------##

#----------------------------------------------##
## fast alignment takes about 10min
CUDA_VISIBLE_DEVICES=${device} python _fast_align.py --name ${name}_Eval
#----------------------------------------------##

#----------------------------------------------##
CUDA_VISIBLE_DEVICES=${device} python test_psnr_ssim.py --save_name ${name}_Eval --bool_crop_and_align False --fake_pth "../talking_head_testing/25fps_video/align_crop/${name}_Eval/*.mp4"
#----------------------------------------------##

#----------------------------------------------##
# test fid
## require more than 3GB GPU
CUDA_VISIBLE_DEVICES=${device} python test_fid.py --save_name ${name}_Eval --fake_pth "../talking_head_testing/25fps_video/align_crop/${name}_Eval/*.mp4" 
#----------------------------------------------##

#----------------------------------------------##
# cd "/home/yxh/yxh_files/benchmark/LMD"
CUDA_VISIBLE_DEVICES=${device} python test_lmd.py --save_name ${name}_Eval --fake_pth "../talking_head_testing/25fps_video/align_crop/${name}_Eval/*.mp4" 
#----------------------------------------------##

#----------------------------------------------##
### sync
CUDA_VISIBLE_DEVICES=${device} python test_sync_conf.py --save_name ${name}_Eval --fake_pth "../talking_head_testing/25fps_video/pcavs_crop/${name}_Eval/*.mp4" --tmp_dir temps/lastversion/${name}_Eval --log_rt results_lastversion 
#----------------------------------------------##

##----------------------------------------------##
### out of memory, require more than 3GB GPU
# CUDA_VISIBLE_DEVICES=${device} python test_emotion_acc.py --in_vid_path "../result/${name}" --save_name ${name}_Eval --emo_range_l 5 --emo_range_r 8 --gpu_id ${device} # test gt
CUDA_VISIBLE_DEVICES=${device} python test_emotion_acc.py --in_vid_path "../result/${name}" --save_name ${name}_Eval --emo_range_l 10 --emo_range_r 13 --gpu_id ${device} # test eat
##----------------------------------------------##

cd "../"
echo " ----------------------------------------------------- " >> res
echo " ${name}" >> res
tail -n 5 "./code/results_lastversion/${name}_Eval.txt" >> res # sync
echo -e "\n" >> res
tail -n 5 "./code/result_psnr/${name}_Eval.txt" >> res # PSNR
echo -e "\n" >> res
tail -n 5 "./code/results/${name}_Eval.txt" >> res # FID
echo -e "\n" >> res
tail -n 8 "./code/result/${name}_Eval.txt" >> res # LMD
echo -e "\n" >> res
tail -n 10 "./code/result_emoacc/${name}_Eval.txt" >> res # EMO_Acc
echo -e "\n" >> res
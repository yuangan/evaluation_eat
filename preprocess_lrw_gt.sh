VIDEO_PATH=${1} # eg. /data3/lipread_test_25/video

cd './code'
python preprocess_lrw.py --save_name lrw_gt --fake_pth ${VIDEO_PATH} --name_mode 1 --need_align_crop #--ours_filter_100


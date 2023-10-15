python preprocess.py  --save_name gt_filter100 --fake_pth '/data4/new_mead/video_test/*.mp4' --name_mode 2 --ours_filter_100
python preprocess.py --save_name mit_filter100 --fake_pth '/data4/makeittalk_othsrc/res_vids/*.mp4' --name_mode 1 --ours_filter_100
python preprocess.py --save_name atvg_filter100 --fake_pth '/data4/ATVG_othsrc_res/res_vids/*.mp4'  --name_mode 2 --ours_filter_100
python preprocess.py --save_name wav2lip_filter100 --fake_pth '/data4/talking_head_testing/temp_res/wav2lip/test_0723/*.mp4'  --name_mode 2 --ours_filter_100
python preprocess.py --save_name pcavs_filter100 --fake_pth '/data4/talking_head_testing/temp_res/pc_avs/temp_0726/*/G_Pose_Driven_.mp4'  --name_mode 6 --ours_filter_100
python preprocess.py --save_name audio2head_filter100 --fake_pth '/data4/talking_head_testing/temp_res/audio2head/test_0722/*.mp4'  --name_mode 2 --ours_filter_100
python preprocess.py --save_name aaai22_filter100 --fake_pth '/home/gy/gy/benchmark/AAAI22-one-shot-talking-face/test_AAAI_MEAD_gtpose/*.mp4' --name_mode 4 --ours_filter_100


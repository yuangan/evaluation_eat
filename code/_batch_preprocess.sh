python preprocess.py --save_name mit_1021 --fake_pth '/data4/talking_head_testing/temp_res/makeittalk/1021/*.mp4' --name_mode 1 --need_align_crop
python preprocess.py --save_name atvg_1021 --fake_pth '/data4/talking_head_testing/temp_res/atvg/1021/*.mp4'  --name_mode 2 --need_align_crop
python preprocess.py --save_name wav2lip_1021 --fake_pth '/data4/talking_head_testing/temp_res/wav2lip/test_1021/*.mp4'  --name_mode 2 --need_align_crop
python preprocess.py --save_name audio2head_1021 --fake_pth '/data4/talking_head_testing/temp_res/audio2head/test_1021/*.mp4'  --name_mode 2 --need_align_crop
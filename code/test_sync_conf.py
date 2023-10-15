#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess, tqdm

from SyncNetInstance import *

# ==================== LOAD PARAMS ====================


# parser = argparse.ArgumentParser(description = "SyncNet");


# opt = parser.parse_args();


# ==================== RUN EVALUATION ====================

def get_parse():
    parser = argparse.ArgumentParser('sync')
    parser.add_argument('--save_name',type=str)
    parser.add_argument('--log_rt',type=str,default='result')
    parser.add_argument('--fake_pth',type=str)
    parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
    parser.add_argument('--batch_size', type=int, default='20', help='');
    parser.add_argument('--vshift', type=int, default='15', help='');
    # parser.add_argument('--videofile', type=str, default="data/example.avi", help='');
    parser.add_argument('--tmp_dir', type=str, default="temps", help='');
    parser.add_argument('--reference', type=str, default="demo", help='');
    return parser


if __name__ == '__main__':

    args = get_parse().parse_args()
    save_name = args.save_name
    fake_pth = args.fake_pth

    args.tmp_dir = args.tmp_dir + '/' + save_name

    s = SyncNetInstance();
    s.loadParameters(args.initial_model);
    print("Model %s loaded."%args.initial_model);

    vids = glob.glob(fake_pth)
    print(len(vids))

    log_rt = args.log_rt
    os.makedirs(log_rt,exist_ok=True)

    fg = open('{}/{}.txt'.format(log_rt,save_name) , 'w')

    conf_lis = []

    id = 0
    for vid_p in tqdm.tqdm(vids):
        id += 1
        # os.rmdir(args.tmp_dir)
        os.system('rm -drf {}'.format(args.tmp_dir))
        offset, conf, dist = s.evaluate(args, videofile=vid_p)

        if offset is None:
            fg.write('{}: no frames\n'.format(vid_p))
            continue

        fg.write('{}:{} {} {}\n'.format(vid_p,conf,offset,dist))
        conf_lis.append(conf)

        if id % 10 == 0 :
            fg.write('pre_avg conf : {}\n'.format( numpy.array(conf_lis).mean() ))

        fg.flush()

    fg.write('avg conf : {}\n'.format( numpy.array(conf_lis).mean() ))
    fg.close()

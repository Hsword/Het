CUDA_VISIBLE_DEVICES=0 python tf_launch_worker.py --model wdl_criteo --config settings/tf_dist_s4_w1.json --rank 0 &
wait

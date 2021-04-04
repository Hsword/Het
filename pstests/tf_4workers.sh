rm -f logs/temp*.log
CUDA_VISIBLE_DEVICES=0 python tf_launch_worker.py --model wdl_criteo --config settings/tf_dist_s4_w4.json --rank 0 > logs/temp0.log &
CUDA_VISIBLE_DEVICES=1 python tf_launch_worker.py --model wdl_criteo --config settings/tf_dist_s4_w4.json --rank 1 > logs/temp1.log &
CUDA_VISIBLE_DEVICES=2 python tf_launch_worker.py --model wdl_criteo --config settings/tf_dist_s4_w4.json --rank 2 > logs/temp2.log &
CUDA_VISIBLE_DEVICES=3 python tf_launch_worker.py --model wdl_criteo --config settings/tf_dist_s4_w4.json --rank 3 > logs/temp3.log &
wait
line0=$(grep 'tensorflow' logs/temp0.log)
line1=$(grep 'tensorflow' logs/temp1.log)
line2=$(grep 'tensorflow' logs/temp2.log)
line3=$(grep 'tensorflow' logs/temp3.log)
exp="print((${line0:13}+${line1:13}+${line2:13}+${line3:13})/4)"
echo $exp
python -c $exp

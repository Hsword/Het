rm -f logs/temp*.log
CUDA_VISIBLE_DEVICES=0 python tf_launch_worker.py --model wdl_criteo --config settings/tf_dist_s4_w2.json --rank 0 > logs/temp0.log & 
CUDA_VISIBLE_DEVICES=1 python tf_launch_worker.py --model wdl_criteo --config settings/tf_dist_s4_w2.json --rank 1 > logs/temp1.log & 
wait
line0=$(grep 'tensorflow' logs/temp0.log)
line1=$(grep 'tensorflow' logs/temp1.log)
exp="print((${line0:13}+${line1:13})/2)"
echo $exp
python -c $exp

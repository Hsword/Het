rm -f logs/temp*.log
CUDA_VISIBLE_DEVICES=0 python tf_launch_worker.py --model wdl_criteo --config settings/tf_dist_s4_w8.json --rank 0 > logs/temp0.log &
CUDA_VISIBLE_DEVICES=1 python tf_launch_worker.py --model wdl_criteo --config settings/tf_dist_s4_w8.json --rank 1 > logs/temp1.log &
CUDA_VISIBLE_DEVICES=2 python tf_launch_worker.py --model wdl_criteo --config settings/tf_dist_s4_w8.json --rank 2 > logs/temp2.log &
CUDA_VISIBLE_DEVICES=3 python tf_launch_worker.py --model wdl_criteo --config settings/tf_dist_s4_w8.json --rank 3 > logs/temp3.log &
CUDA_VISIBLE_DEVICES=4 python tf_launch_worker.py --model wdl_criteo --config settings/tf_dist_s4_w8.json --rank 4 > logs/temp4.log &
CUDA_VISIBLE_DEVICES=5 python tf_launch_worker.py --model wdl_criteo --config settings/tf_dist_s4_w8.json --rank 5 > logs/temp5.log &
CUDA_VISIBLE_DEVICES=6 python tf_launch_worker.py --model wdl_criteo --config settings/tf_dist_s4_w8.json --rank 6 > logs/temp6.log &
CUDA_VISIBLE_DEVICES=7 python tf_launch_worker.py --model wdl_criteo --config settings/tf_dist_s4_w8.json --rank 7 > logs/temp7.log &
wait
line0=$(grep 'tensorflow' logs/temp0.log)
line1=$(grep 'tensorflow' logs/temp1.log)
line2=$(grep 'tensorflow' logs/temp2.log)
line3=$(grep 'tensorflow' logs/temp3.log)
line4=$(grep 'tensorflow' logs/temp4.log)
line5=$(grep 'tensorflow' logs/temp5.log)
line6=$(grep 'tensorflow' logs/temp6.log)
line7=$(grep 'tensorflow' logs/temp7.log)
exp="print((${line0:13}+${line1:13}+${line2:13}+${line3:13}+${line4:13}+${line5:13}+${line6:13}+${line7:13})/8)"
echo $exp
python -c $exp

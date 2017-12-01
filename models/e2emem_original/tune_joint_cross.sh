L2=( 0.001 0.00001 0.000001 0.0000001 )
ms=( 50 75 130 )

dir=babi/tasks_1-20_v1-2/en/

for lambda in "${L2[@]}"; do
    for m in "${ms[@]}"; do
        echo tuning $lambda $m
        python joint.py --data_dir $dir --output_file plogs/scores_ms_${m}_lambda_${lambda}.csv --early 50 --learning_rate 0.001 --memory_size $m --regularization $lambda |& tee plogs/joint_cross_ms_${m}_lambda_${lambda}.txt
    done
done

L2=( 3e-7 3e-6 3e-5 3e-4 3e-3 3e-2 1e-1 3e-1 )

alpha=0.001
dir=../cs224d-project/data/babi/tasks_1-20_v1-2/en/

for lamb in "${L2[@]}"; do
    echo tuning $lamb
    python joint.py --data_dir $dir --output_file scores_alpha_$alpha.lambda_$lamb.csv --early 50 --learning_rate $alpha --regularization $lamb > joint_alpha_0.001_lambda_$lamb.txt
done

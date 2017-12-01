lr=( 0.01 0.05 0.005 )
hp=( 3 6 )
es=( 30 40 )
ms=( 50 60 70 )

task=2

for lrn in "${lr[@]}"; do
    for hop in "${hp[@]}"; do
        for ems in "${es[@]}"; do
            # for mems in "${ms[@]}"; do
                echo $lrn, $hop, $ems, $mems
                python single.py --task $task --learning_rate $lrn --hops $hop --embedding_size $ems #--memory_size $mems
            # done
        done
    done
done

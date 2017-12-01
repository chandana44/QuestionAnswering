for i in {1..20}; do
    echo $i
    python single.py --task_id $i
done

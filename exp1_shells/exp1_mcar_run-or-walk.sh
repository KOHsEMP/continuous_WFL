#!/bin/bash

exp_name='continuous_exp1'
base_name='exp_run-or-walk_base'

config_dir='./config'
base_yaml="${config_dir}/${base_name}.yaml"


miss_pattern='MCAR'

rm_k=2

seed_list=(42 43 44 45 46)
use_train_size_list=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000)
miss_rate_list=(0.25 0.5 0.75)
est_error_mse_list=(0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5)

for seed in ${seed_list[@]}
do
    for miss_rate in ${miss_rate_list[@]}
    do
        for est_error_mse in ${est_error_mse_list[@]}
        do
            for use_train_size in ${use_train_size_list[@]}
            do
            tmp_yaml="${config_dir}/tmp_${base_name}_${exp_name}_${miss_pattern}_${miss_rate}_${use_train_size}_${est_error_mse}_${rm_k}_${seed}.yaml"
            cp ${base_yaml} ${tmp_yaml}

            echo "" >> ${tmp_yaml}
            echo "exp_name: ${exp_name}" >> ${tmp_yaml}
            echo "miss_pattern: ${miss_pattern}" >> ${tmp_yaml}
            echo "rm_k: ${rm_k}" >> ${tmp_yaml}

            echo "seed: ${seed}" >> ${tmp_yaml}
            echo "miss_rate: ${miss_rate}" >> ${tmp_yaml}
            echo "use_train_size: ${use_train_size}" >> ${tmp_yaml}
            echo "est_error_mse: ${est_error_mse}" >> ${tmp_yaml}

            python ./exp1.py --config_file ${tmp_yaml}
            rm ${tmp_yaml}
            done
        done
    done
done
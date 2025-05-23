# set -e

export CUDA_VISIBLE_DEVICES=7

model_name=unitsf
seed=2024
model_id=Env_96_48
path=./dataset/mmd/
file=Environment.csv
exp_name=env_48_s
data=custom
setting=S
seq_len=96
pred_len=48
label_len=48
# d_model=512
d_ff=2048
n_heads=4
e_layers=2
enc_in=1
dec_in=1
c_out=1

use_norm_values=("False" "True")
use_decomp_values=("False" "True")
fusion_values=("temporal" "feature")
emb_type_values=("token" "patch" "invert" "freq" "none")
ff_type_values=("mlp" "rnn" "trans")


for use_norm in "${use_norm_values[@]}"; do
    for use_decomp in "${use_decomp_values[@]}"; do
        for fusion in "${fusion_values[@]}"; do
            for emb_type in "${emb_type_values[@]}"; do
                for ff_type in "${ff_type_values[@]}"; do

                    if [[ "$ff_type" == "rnn" ]]; then
                        d_model=256
                    else
                        d_model=512
                    fi

                    echo "Running with use_norm=$use_norm, use_decomp=$use_decomp, fusion=$fusion, emb_type=$emb_type, ff_type=$ff_type, d_model=$d_model"

                    python -u run.py --seed $seed --task_name long_term_forecast --model $model_name --model_id $model_id --is_training 1 --root_path $path --data_path $file --data $data --features $setting \
                            --seq_len $seq_len --pred_len $pred_len --label_len $label_len --use_norm $use_norm --use_decomp $use_decomp --fusion $fusion --emb_type $emb_type --ff_type $ff_type \
                            --d_model $d_model --n_heads $n_heads --e_layers $e_layers --d_ff $d_ff --enc_in $enc_in --dec_in $dec_in --c_out $c_out \
                            --exp_name $exp_name

                done
            done
        done
    done
done
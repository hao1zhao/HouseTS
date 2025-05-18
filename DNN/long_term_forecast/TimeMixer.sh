export CUDA_VISIBLE_DEVICES=0

for combo in "6 3" "6 6" "6 12" "12 3" "12 6" "12 12"; do
  read seq_len pred_len <<< "${combo}"
  model_id="house_TSMixer_sl${seq_len}_pl${pred_len}"

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id "${model_id}" \
    --model TSMixer \
    --data custom \
    --freq ym \
    --root_path ./dataset/ \
    --data_path HouseTS_log.csv \
    --features MS \
    --target price \
    --seq_len "${seq_len}" \
    --label_len 3 \
    --pred_len "${pred_len}" \
    --e_layers 2 \
    --enc_in 33 \
    --c_out 1 \
    --d_model 128 \
    --learning_rate 0.0001 \
    --num_workers 4 \
    --train_epochs 10
done


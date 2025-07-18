# nsys profile \
#     --trace=cuda,cublas,osrt,nvtx \
#     --gpu-metrics-devices=all \
#     --cuda-memory-usage=true \
#     --force-overwrite=true \
#     --output  flash_attn_profile pytest ./tests/test_flash_attn.py -k test_flash_attn_varlen_output -s

nsys profile \
    --trace=cuda,cublas,osrt,nvtx \
    --gpu-metrics-devices=all \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --output  flash_attn_nonvar_profile pytest ./tests/test_flash_attn.py -k test_flash_attn_output_multiple_runs -s

# ncu --target-processes all \
#     --kernel-name ""flash_fwd_kernel \
#     --set full \
#     --export flash_attn_ncu_profile \
#     pytest ./tests/test_flash_attn.py -k test_flash_attn_varlen_output -s


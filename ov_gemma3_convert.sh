export HF_ENDPOINT=https://hf-mirror.com

# optimum-cli export openvino --model google/gemma-3-4b-it --task image-text-to-text ov-gemma-3-4b-it

optimum-cli export openvino --model google/gemma-3-4b-it --task image-text-to-text --weight-format int4  ov-gemma-3-4b-it-INT4
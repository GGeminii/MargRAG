# Fill these in:
export OPENAI_API_KEY="your api key"
export MINERU_PATH="/../anaconda3/envs/mineru/bin"   # folder that contains 'magic-pdf'
export OPENAI_BASE_URL=
export OPENAI_MODEL=
# Download the embedded model for the first time and set up the image
export HF_ENDPOINT="https://hf-mirror.com"
# If the embedded model has been downloaded locally,/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct, temporarily cancel the mirroring and use the local model instead
#unset HF_ENDPOINT
#export HF_HUB_OFFLINE=1  # 开启离线模式
#export HF_HUB_CACHE=/home/ubuntu/.cache/huggingface/hub  # 指定缓存目录
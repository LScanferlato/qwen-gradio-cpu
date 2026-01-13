podman build -t qwen-gradio-cpu .

podman run --rm \
    -p 7860:7860 \
    -v $(pwd)/hf_cache:/hf_cache \
    qwen-gradio-cpu

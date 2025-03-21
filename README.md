# OpenVINO enable Gemma3 multimodal model

Requirement : Init environment 
```
pip install -r requirements.txt
```
- Step 1. HF model downloading 

```
python hf_gemma3_download.py
# Need to use HF access token for the code below to run.
```
- Step 2. OpenVINO model convert 
```
./ov_gemma3_convert.sh
```
- Step 3. OpenVINO model inference
```
python ov_gemma3_infer.py
```
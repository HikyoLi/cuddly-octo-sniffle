# 使用FFmpeg提取帧（25fps）
ffmpeg -i input.avi -vf "fps=25,scale=512:512" frames/%04d.png

# 人脸检测与对齐（使用MediaPipe）
python -m mediapipe.examples.python.face_detection \
  --input_face_image=frames/0001.png \
  --output_cropped_face=aligned_faces/0001.png

  from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
audio_input = processor(audio, sampling_rate=16000, return_tensors="pt")

# 提取音素对齐（使用Montreal Forced Aligner）
mfa align ./audio ./lexicon.txt chinese_mfa ./output

{
  "dataset": {
    "camera_angle_x": 0.857,  # 通过相机标定获取
    "frames": [
      {
        "file_path": "aligned_faces/0001",
        "rotation": 0.0,      # 用于视角增强
        "audio_features": [0.12, -0.45, ...],  # Wav2Vec2特征
        "phonemes": ["sil", "sh", "i", ...]     # MFA对齐结果
      }
    ]
  }
}

model:
  nerf:
    audio_dim: 1024           # Wav2Vec2特征维度
    num_phonemes: 43          # 中文音素数（根据MFA调整）
  render:
    num_samples: 64           # 光线采样数

train:
  lr: 5e-4
  num_epochs: 2000
  batch_size: 8               



python train.py \
  --config configs/cmlr.yaml \
  --data_dir ./processed_cmlr \
  --exp_name cmlr_radnerf \
  --aud_file ./audio_features/wav2vec2.pt


# 在radnerf/utils/phoneme_utils.py中修改中文映射
PHONEME_MAP = {
    "sil": 0,    # 静音
    "sh": 1,     # ʂ (中文"诗")
    "zh": 2,     # tʂ ("知")
    # ...其他43个中文音素
}


# 在radnerf/model/audio_encoder.py中添加中文韵律特征
def extract_prosody(audio):
    pitch = parselmouth.Sound(audio).to_pitch()
    intensity = parselmouth.Sound(audio).to_intensity()
    return torch.cat([pitch.values, intensity.values], dim=-1)


tensorboard --logdir ./logs/cmlr_radnerf

# 每隔100epoch生成测试样本
python render.py \
  --config configs/cmlr.yaml \
  --checkpoint ./logs/cmlr_radnerf/checkpoint_100.pth \
  --audio test_audio.wav

# 在train.py中添加
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    outputs = model(batch)
    loss = compute_loss(outputs)
scaler.scale(loss).backward()
scaler.step(optimizer)


# 使用2块GPU训练
python -m torch.distributed.launch --nproc_per_node=2 train.py ...


# 导出为TorchScript
traced_model = torch.jit.trace(model, example_inputs)
traced_model.save("radnerf_cmlr.pt")

from fastapi import FastAPI
app = FastAPI()

@app.post("/generate")
async def generate(audio: UploadFile):
    audio_data = await audio.read()
    features = extract_features(audio_data)
    render_video = model(features)
    return StreamingResponse(render_video, media_type="video/mp4")




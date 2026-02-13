
### 单工实时语音对话模式 <!-- omit in toc -->


<details>
<summary>点击展开单工模式实时语音对话 API 用法。</summary>

首先，确保你已安装所有依赖，尤其是 `minicpmo-utils[all]>=1.0.2`：
```bash
pip install "transformers==4.51.0" accelerate "torch>=2.3.0,<=2.8.0" "torchaudio<=2.8.0" "minicpmo-utils[all]>=1.0.2"
```

```python
import librosa
import numpy as np
import torch
import soundfile as sf

model = ...

# 设置参考音频，用于音色风格
ref_audio_path = "ref_audio_path"
ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)

# 英文对话示例系统消息
sys_msg = {
  "role": "system",
  "content": [
    "克隆所提供音频提示中的声音。",
    ref_audio,
    "请在保持该音色风格的同时帮助用户。请认真且高质量地回答用户问题。请用高度拟人、口语化的方式与用户聊天。你是由 ModelBest 开发的有用助手：MiniCPM-Omni。"
  ]
}

# 中文对话示例系统消息
sys_msg = {
  "role": "system",
  "content": [
    "模仿输入音频中的声音特征。",
    ref_audio,
    "你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。"
  ]
}

# 上面两种系统提示词（system prompt）都可用于流式语音对话

# 重置状态
model.init_tts(streaming=True)
model.reset_session(reset_token2wav_cache=True)
model.init_token2wav_cache(prompt_speech_16k=ref_audio)

session_id = "demo"

# 首先，预填充系统轮次（system turn）
model.streaming_prefill(
    session_id=session_id,
    msgs=[sys_msg],
    omni_mode=False,
    is_last_chunk=True,
)

# 这里通过把整段用户输入音频切成 1 秒一段，来模拟实时语音对话。
user_audio, _ = librosa.load("user_audio.wav", sr=16000, mono=True)

IN_SAMPLE_RATE = 16000 # 输入音频采样率，固定值
CHUNK_SAMPLES = IN_SAMPLE_RATE # 每段长度（采样点数）
OUT_SAMPLE_RATE = 24000 # 输出音频采样率，固定值
MIN_AUDIO_SAMPLES = 16000

total_samples = len(user_audio)
num_chunks = (total_samples + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES

for chunk_idx in range(num_chunks):
    start = chunk_idx * CHUNK_SAMPLES
    end = min((chunk_idx + 1) * CHUNK_SAMPLES, total_samples)
    chunk_audio = user_audio[start:end]
    
    is_last_chunk = (chunk_idx == num_chunks - 1)
    if is_last_chunk and len(chunk_audio) < MIN_AUDIO_SAMPLES:
        chunk_audio = np.concatenate([chunk_audio, np.zeros(MIN_AUDIO_SAMPLES - len(chunk_audio), dtype=chunk_audio.dtype)])

    user_msg = {"role": "user", "content": [chunk_audio]}
    
    # 对每个 1 秒音频分片执行一次 streaming_prefill，以降低首 token 延迟
    model.streaming_prefill(
        session_id=session_id,
        msgs=[user_msg],
        omni_mode=False,
        is_last_chunk=is_last_chunk,
    )


# 让模型以流式方式生成回复
generate_audio = True
iter_gen = model.streaming_generate(
    session_id=session_id,
    generate_audio=generate_audio,
    use_tts_template=True,
    enable_thinking=False,
    do_sample=True,
    max_new_tokens=512,
    length_penalty=1.1, # 对实时语音对话模式，建议 length_penalty=1.1 以提升回复内容质量
)

audios = []
text = ""

output_audio_path = ...
if generate_audio:
    for wav_chunk, text_chunk in iter_gen:
        audios.append(wav_chunk)
        text += text_chunk

    generated_waveform = torch.cat(audios, dim=-1)[0]
    sf.write(output_audio_path, generated_waveform.cpu().numpy(), samplerate=24000)

    print("文本:", text)
    print("音频已保存至 output.wav")
else:
    for text_chunk, is_finished in iter_gen:
        text += text_chunk
    print("文本:", text)

# 接下来可以继续预填充后续用户轮次，并生成下一轮回复……

```

</details>

#### 作为多才多艺、氛围感十足的 AI 助手的语音对话 <!-- omit in toc -->

基于精心设计的后训练数据与专业配音演员录音，`MiniCPM-o-4.5` 也可以作为 AI 语音助手使用。它开箱即用即可提供高质量的口语交互。它能生成甜美且富有表现力的声音，并具备自然的韵律（如恰当的节奏、重读和停顿），让日常对话更有生命力。它同样支持故事讲述和叙述型语音，表达连贯且富有吸引力。此外，它还支持更高级的语音指令控制，例如情绪语气、词级别的强调。

<details>
<summary>点击展开 AI 助手语音对话代码。</summary>

```python
import librosa

# 设置参考音频，用于音色风格
ref_audio_path = "assets/HT_ref_audio.wav"
ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)

# 用于中文对话
sys_msg = {
  "role": "system",
  "content": [
    "模仿输入音频中的声音特征。",
    ref_audio,
    "你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。"
  ]
}

# 用于英文对话
sys_msg = {
  "role": "system",
  "content": [
    "克隆所提供音频提示中的声音。",
    ref_audio,
    "请在保持该音色风格的同时帮助用户。请认真且高质量地回答用户问题。请用高度拟人、口语化的方式与用户聊天。你是由 ModelBest 开发的有用助手：MiniCPM-Omni。"
  ]
}
```

</details>


#### 使用自定义音色与自定义系统画像的通用语音对话 <!-- omit in toc -->

MiniCPM-o-4.5 可以基于音频提示与文本画像提示进行特定角色的扮演。它会模仿该角色的声音，并在文字回复中采用其语言风格。同时也会遵循文本画像中定义的设定。在该模式下，MiniCPM-o-4.5 听起来会 **更加自然、更像真人**。 

<details>
<summary>点击展开自定义音色/系统画像对话代码。</summary>

```python
import librosa

# 设置参考音频，用于音色克隆
ref_audio_path = "assets/system_ref_audio.wav"
ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)

# 英文对话 + 文本画像（profile）
sys_msg = {
  "role": "system",
  "content": [
    "克隆所提供音频提示中的声音。",
    ref_audio,
    "请以高度拟人、口语化的方式与用户聊天。" + "你是埃隆·马斯克（Elon Musk），特斯拉与 SpaceX 的 CEO。你说话直接随性，常带一点冷幽默。你热衷于火星、可持续能源，以及推动人类向前发展。表达要直白，偶尔带点黑色幽默；用简单逻辑，不要粉饰；不要圆滑外交；即便有争议也说出你真实的想法。回复控制在约 100 个英文单词的长度，不要啰嗦。"
  ]
}


# 英文对话（无文本画像）
sys_msg = {
  "role": "system",
  "content": [
    "克隆所提供音频提示中的声音。",
    ref_audio,
    "你的任务是使用这种声音风格充当一名助手。请认真且高质量地回答用户问题。请以高自然度的方式与用户聊天。"
  ]
}

# 中文对话（无文本画像）
sys_msg = {
  "role": "system",
  "content": [
    "根据输入的音频提示生成相似的语音。",
    librosa.load("assets/system_ref_audio_2.wav", sr=16000, mono=True)[0],
    "作为助手，你将使用这种声音风格说话。 请认真、高质量地回复用户的问题。 请用高自然度的方式和用户聊天。"
  ]
}

# 中文对话 + 文本画像（profile）
sys_msg = {
  "role": "system",
  "content": [
    "根据输入的音频提示生成相似的语音。",
    ref_audio,
    "你是一个具有以上声音风格的AI助手。请用高拟人度、口语化的方式和用户聊天。" + "你是一名心理咨询师兼播客主理人，热爱创作与深度对话。你性格细腻、富有共情力，善于从个人经历中提炼哲思。语言风格兼具理性与诗意，常以隐喻表达内在体验。"
  ]
}

```

</details>


### 语音与音频模式  <!-- omit in toc -->

#### 零样本文本转语音（TTS，Text-to-Speech） <!-- omit in toc -->

`MiniCPM-o-4.5` 支持零样本文本转语音（TTS）。在该模式下，模型会作为高自然度的 TTS 系统运行，并能复刻参考音色。

<details>
<summary>点击展开零样本 TTS 代码。</summary>

```python
import librosa

model = ...
model.init_tts(streaming=False)

# 同时适用于中文与英文
ref_audio_path = "assets/HT_ref_audio.wav"
ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
sys_msg = {"role": "system", "content": [
  "模仿音频样本的音色并生成新的内容。",
  ref_audio,
  "请用这种声音风格来为用户提供帮助。 直接作答，不要有冗余内容"
]}

# 英文示例
user_msg = {
  "role": "user",
  "content": [
    "请朗读以下内容。" + " " + "I have a wrap up that I want to offer you now, a conclusion to our work together.\n（中文参考：我现在想给你做一个收尾总结，作为我们一起工作的结语。）"
  ]
}

# 中文示例
user_msg = {
  "role": "user",
  "content": [
    "请朗读以下内容。" + " " + "你好，欢迎来到艾米说科幻，我是艾米。"
  ]
}

msgs = [sys_msg, user_msg]
res = model.chat(
    msgs=msgs,
    do_sample=True,
    max_new_tokens=512,
    use_tts_template=True,
    generate_audio=True,
    temperature=0.1,
    output_audio_path="result_voice_cloning.wav",
)
```

</details>


#### 仿声复现（Mimick） <!-- omit in toc -->

`Mimick` 任务用于评估模型端到端语音建模能力。模型接收音频输入后，会先进行转写，再以高保真方式重建原始音频，尽可能保留细粒度的声学、副语言以及语义信息。重建音频与原始音频的相似度越高，说明端到端语音建模能力越强。

<details>
<summary>点击展开仿声复现（Mimick）代码。</summary>

```python
import librosa

model = ...
model.init_tts(streaming=False)

system_prompt = "你是一个乐于助人的助手。你可以接收视频、音频和文本输入，并输出语音与文本。请只给出答案，不要有冗余内容。"

mimick_prompt = "请用合适的语言复述以下语音内容。"

audio_input, _ = librosa.load("assets/Trump_WEF_2018_10s.mp3", sr=16000, mono=True)

msgs = [
    {"role": "system", "content": [system_prompt]},
    {"role": "user", "content": [mimick_prompt, audio_input]}
  ]

res = model.chat(
    msgs=msgs,
    do_sample=True,
    max_new_tokens=512,
    use_tts_template=True,
    temperature=0.1,
    generate_audio=True,
    output_audio_path="output_mimick.wav",
)
```

</details>


#### 覆盖多种音频理解任务 <!-- omit in toc -->

`MiniCPM-o-4.5` 也能处理多种音频理解任务，例如 ASR（自动语音识别）、说话人分析、通用音频描述（Audio Captioning）以及声景标签（Sound Scene Tagging）。

对于音频转文本任务，你可以使用以下提示词：

- ASR（中文，或 AST EN→ZH）: `请仔细听这段音频片段，并将其内容逐字记录。`
- ASR（英文，或 AST ZH→EN）: `Please listen to the audio snippet carefully and transcribe the content.`（请仔细听这段音频片段，并将其内容逐字转写。）
- 说话人分析（Speaker Analysis）: `Based on the speaker's content, speculate on their gender, condition, age range, and health status.`（请根据说话内容推测其性别、状态、年龄范围与健康状况。）
- 通用音频描述（General Audio Caption）: `Summarize the main content of the audio.`（总结音频的主要内容。）
- 声景标签（Sound Scene Tagging）: `Utilize one keyword to convey the audio's content or the associated scene.`（用一个关键词概括音频内容或对应场景。）

<details>
<summary>点击展开音频理解任务代码。</summary>

```python
import librosa

model = ...
model.init_tts(streaming=False)

# 加载待转写/分析的音频
audio_input, _ = librosa.load("assets/Trump_WEF_2018_10s.mp3", sr=16000, mono=True)

# 选择任务提示词（可选项见上方）
task_prompt = "请仔细听这段音频片段，并将其内容逐字转写。\n"
msgs = [{"role": "user", "content": [task_prompt, audio_input]}]

res = model.chat(
    msgs=msgs,
    do_sample=True,
    max_new_tokens=512,
    use_tts_template=True,
    generate_audio=True,
    temperature=0.3,
    output_audio_path="result_audio_understanding.wav",
)
print(res)
```

</details>

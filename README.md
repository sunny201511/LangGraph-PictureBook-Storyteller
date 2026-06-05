# 🌙 LangGraph PictureBook Storyteller (智能儿童绘本故事机)

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![LangGraph](https://img.shields.io/badge/Framework-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20%2F%20Qwen2.5-purple.svg)](https://ollama.com)

一个基于 **LangGraph** 多智能体工作流架构开发的**智能儿童绘本故事机**。只需运行一行命令或通过 LangGraph Studio 界面，系统就能自动扫描绘本图片、智能净化纠错 OCR 文本、合成温柔的真人语音，并无损渲染出带有全真字幕的儿童睡前故事视频。

极其适合家中有小宝贝、想把纸质/电子绘本快速转化为有声视频故事的家长，以及想深入学习 LangGraph 生产级状态机架构的开发者。

---

## ✨ 核心特性

- 🏗️ **可靠的状态机架构**：全流程基于 `LangGraph` 的 `StateGraph` 构建，节点职责清晰，数据流严格受控，彻底告别传统 AI 链式调用的不可控性。
- 👁️ **自适应增强 OCR 扫描**：集成 `Tesseract OCR` 引擎，内置画质自适应增强算法（自动灰度化、高对比度降噪、二值化处理），完美攻克绘本排版零散、底纹复杂的识别难题。
- 🧠 **纯净本地双模式驱动**：
  - **AI 净化不扩写模式**：由本地 `Ollama (默认 qwen2.5:3b)` 驱动，仅做去噪、纠错与平缓口语化微调，保留绘本原汁原味，严禁盲目编造剧情。
  - **上下文智能连接**：当某页绘本由于大面积插画未识别出明显文字时，AI 会结合前后文主题自动创作极简剧情衔接句。
- 🎙️ **高质量有声合成**：内置 `Edge-TTS` 神经网络语音（默认温柔安抚的 `zh-CN-XiaoxiaoNeural`），支持自适应语速调节（0.85x 慢速慢读，更适合幼儿倾听）。
- 🎬 **硬件加速视频合成**：支持 Mac 系统的 `VideoToolbox` 硬件加速，一键秒级生成高画质、带精美半透明动态字幕的 `MP4` 睡前故事视频。

---

## 📐 状态机工作流架构 (Workflow Graph)

项目完全采用循环/线性可扩展的图结构，标准运行节点如下：


```

[init] (初始化与图片预处理)
│
[analyze_images] (Tesseract OCR 画质增强扫描)
│
[generate_story] (Ollama 文本去噪与纯净纠错)
│
[synthesize_audio] (Edge-TTS 动态音频合成)
│
[render_frames] (离线全真半透明字幕绘制)
│
[compose_video] (FFmpeg 硬件加速视频合成)
│
[save_output] (保存结构化清单 JSON 并结束) ➡️ [END]

```

---

## 🛠️ 快速开始（本地部署）

### 1. 环境准备
确保你的 Mac（推荐）或 Linux 系统上已经安装了以下底层工具：
```bash
# 安装 FFmpeg (处理视音频) 和 Tesseract (提供 OCR 能力)
brew install ffmpeg tesseract tesseract-lang

# 启动并下载本地大模型 (推荐 qwen2.5:3b，轻量且中文精炼)
ollama run qwen2.5:3b

```

### 2. 克隆项目并安装依赖

```bash
git clone [https://github.com/sunny201511/LangGraph-PictureBook-Storyteller.git](https://github.com/sunny201511/LangGraph-PictureBook-Storyteller.git)
cd LangGraph-PictureBook-Storyteller

# 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

```

### 3. 配置环境变量

在项目根目录下新建 `.env` 文件，用于启用可选的 LangSmith 链路追踪（项目已做双向兼容，不配置也不影响本地核心流程跑通）：

```ini
LANGSMITH_API_KEY="你的LangSmith密钥"
LANGSMITH_TRACING="true"
LANGCHAIN_PROJECT="my-langgraph-storyteller"

```

---

## 🚀 运行与使用

### 方式 A：命令行一键生成

将你需要处理的绘本图片（支持 `.jpg/.png/.webp`）放入 `./images/` 目录，然后直接在虚拟环境中运行：

```bash
python bedtime_story_langgraph.py --images ./images/ --theme "聪明的小狐狸" --age "3-6岁"

```

**常用命令行参数说明：**

* `--images`: 绘本图片源目录 (默认 `./images/`)
* `--theme`: 绘本的主线全局基调主题
* `--age`: 目标儿童年龄段 (如 `3-6岁`)
* `--model`: 本地 Ollama 模型名称 (默认 `qwen2.5:3b`)
* `--speed`: 故事播放语速 (默认 `0.85` 慢速安抚)

### 方式 B：LangGraph Studio 可视化调试

本项目无缝适配了官方的 `LangGraph Studio` 视觉面板。

1. 在子目录中创建好了符合规范的 `langgraph.json`。
2. 打开 `LangGraph Studio` 软件，选择 `my-langgraph-storyteller` 文件夹导入。
3. 在可视化的 Graph 界面中，你可以直观地看到每一个节点的流转状态、输入输出明细，极大地方便了多智能体的调试与扩展。

---

## 📂 输出成果展示

运行完成后，系统会在指定的输出目录下生成以下资产：

```text
output/
├── images/           # 经过标准化比例裁切的绘本图片
├── audio/            # 逐页生成的、带有时间戳的高音质 MP3 音频
├── frames/           # 离线叠加了半透明字幕的视频帧图片
└── video/
    └── story_video.mp4  # 🌟 最终可以直接在投影仪或平板上给宝宝播放的故事视频

```

---

## 🤝 参与贡献与社区活跃

非常欢迎感兴趣的开发者加入，共同丰富儿童智能交互的生态！你可以通过以下方式积极参与：

1. **提交 Issue**：反馈你遇到的 Tesseract 识别语种、特定绘本排版下的去噪效果问题。
2. **提交 PR**：目前项目预留了多 Agent 扩展接口。欢迎贡献如 `Stable Diffusion 绘本插画重绘 Agent`、`情绪背景音乐自动编排 Agent` 等新节点。

*宝贝，晚安。* 🌙

```

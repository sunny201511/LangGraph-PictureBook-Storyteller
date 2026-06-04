import os
from pathlib import Path

# ═════════════ 动态加载 .env 配置文件 ═════════════
try:
    from dotenv import load_dotenv
    # 指定加载当前文件所在目录下的 .env 文件
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    # 兼容未安装 python-dotenv 的极端情况
    pass

# ═════════════ 自动激活 LangSmith 追踪 ═════════════
# 优先从 .env 读取，如果读取不到则从你的截图中同步对应的 key 名
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "my-langgraph-storyteller")
# ════════════════════════════════════════════════


import asyncio
import json
import shutil
import argparse
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TypedDict, List

from langgraph.graph import StateGraph, END
import edge_tts
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance
import pytesseract


# ══════════════════════════════════════════════════════════════════════════════
# 1. 状态定义
# ══════════════════════════════════════════════════════════════════════════════

class StoryState(TypedDict):
    image_paths: List[str]
    raw_image_paths: List[str] # 保存原始高分辨率图片路径，专供 OCR 使用
    output_dir: str
    theme: str
    age_group: str
    story_style: str
    voice_name: str
    voice_speed: float
    resolution: tuple          # (w, h)
    page_duration: float       # 0 = 随音频
    ollama_model: str          # 默认 qwen2.5:3b
    image_descriptions: List[str] 
    story_texts: List[str]
    audio_paths: List[str]
    audio_durations: List[float]
    frame_paths: List[str]     
    video_path: str
    story_json_path: str
    status: str


# ══════════════════════════════════════════════════════════════════════════════
# 2. 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def setup_dirs(output_dir: str) -> dict:
    root = Path(output_dir)
    dirs = {
        "root": root,
        "images": root / "images",
        "frames": root / "frames",   
        "audio":  root / "audio",
        "video":  root / "video",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return {k: str(v) for k, v in dirs.items()}


def resize_image(src: str, dst: str, w: int, h: int):
    img = Image.open(src).convert("RGB")
    src_w, src_h = img.size
    scale = max(w / src_w, h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - w) // 2
    top  = (new_h - h) // 2
    img = img.crop((left, top, left + w, top + h))
    img.save(dst, "JPEG", quality=88, optimize=True)


def draw_subtitle(img_path: str, dst_path: str, text: str, w: int, h: int):
    font_candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/Library/Fonts/Arial Unicode MS.ttf",
    ]
    font_size = max(26, h // 22)
    font = None
    for fp in font_candidates:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                continue

    img = Image.open(img_path).convert("RGB")
    if font is None or not text.strip():
        img.save(dst_path, "JPEG", quality=88)
        return

    draw = ImageDraw.Draw(img)
    max_chars = max(10, int(w * 0.82 / font_size))
    
    lines = []
    while len(text) > max_chars:
        lines.append(text[:max_chars])
        text = text[max_chars:]
    lines.append(text)

    line_h = font_size + 8
    total_h = line_h * len(lines) + 16
    bar_top = h - total_h - 20

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([(0, bar_top), (w, h)], fill=(0, 0, 0, 160))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    y = bar_top + 8
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        lw = bbox[2] - bbox[0]
        x = (w - lw) // 2
        for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
            draw.text((x+dx, y+dy), line, font=font, fill=(0,0,0))
        draw.text((x, y), line, font=font, fill=(255,255,255))
        y += line_h

    img.save(dst_path, "JPEG", quality=88)


def ollama_chat(model: str, prompt: str, timeout: int = 120) -> str:
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=model, temperature=0.7)
        resp = llm.invoke(prompt)
        return resp.content.strip()
    except Exception:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()


def timer(label: str):
    class T:
        def __enter__(self):
            self.t = time.time()
            print(f"\n⏱  {label}...")
            return self
        def __exit__(self, *_):
            print(f"   完成，耗时 {time.time()-self.t:.1f}s")
    return T()


# ══════════════════════════════════════════════════════════════════════════════
# 3. LangGraph 节点
# ══════════════════════════════════════════════════════════════════════════════

def init_node(state: StoryState) -> StoryState:
    with timer("节点1 · 初始化 & 图片预处理"):
        dirs = setup_dirs(state["output_dir"])
        w, h = state["resolution"]

        image_paths = sorted(state["image_paths"])
        saved_paths = []
        raw_saved_paths = []
        
        for i, src in enumerate(image_paths):
            # 备份原始高无损分辨率路径，防止后面剪裁缩放破坏 OCR 清晰度
            raw_saved_paths.append(src)
            
            dst = Path(dirs["images"]) / f"page_{i+1:02d}.jpg"
            resize_image(src, str(dst), w, h)
            saved_paths.append(str(dst))
            print(f"   图片 {i+1}/{len(image_paths)}: {Path(src).name} → {dst.name}")

    return {
        **state,
        "image_paths": saved_paths,
        "raw_image_paths": raw_saved_paths, 
        "image_descriptions": [],
        "story_texts": [],
        "audio_paths": [],
        "audio_durations": [],
        "frame_paths": [],
        "video_path": "",
        "story_json_path": "",
        "status": "initialized",
    }


def analyze_images_node(state: StoryState) -> StoryState:
    with timer("节点2 · Tesseract OCR 提取原图文字（自适应画质增强版）"):
        descriptions = []
        lang_config = "chi_sim+eng"  

        # 优先读取原始高清无损图进行识别
        target_paths = state.get("raw_image_paths", state["image_paths"])

        for i, img_path in enumerate(target_paths):
            detected_text = ""
            try:
                img = Image.open(img_path)
                img_gray = ImageOps.grayscale(img)
                
                # 图像增强：放大对比度，消除纸张泛黄和复杂的绘本背景底纹
                img_enhanced = ImageEnhance.Contrast(img_gray).enhance(3.0)     
                img_enhanced = ImageEnhance.Brightness(img_enhanced).enhance(1.1) 
                img_thresh = img_enhanced.point(lambda p: 255 if p > 135 else 0)
                
                # 模式一：--psm 11（稀疏文本模式，精准捕捉零散、不规则排版的绘本文字）
                custom_config = r'--psm 11'
                raw_text = pytesseract.image_to_string(img_thresh, lang=lang_config, config=custom_config)
                detected_text = " ".join(raw_text.split()).strip()
                
                # 模式二兜底：若没抓到，换用 --psm 6 重新扫描
                if not detected_text:
                    custom_config_backup = r'--psm 6'
                    raw_text = pytesseract.image_to_string(img_thresh, lang=lang_config, config=custom_config_backup)
                    detected_text = " ".join(raw_text.split()).strip()
                    
            except Exception as e:
                print(f"   ❌ Tesseract 识别第 {i+1} 页失败: {e}")

            if not detected_text:
                detected_text = f"（本页未识别出明显文字，请根据整体主题《{state['theme']}》合理发挥第 {i+1} 页剧情）"

            desc = f"第 {i+1} 页原图文字线索：{detected_text}"
            descriptions.append(desc)
            print(f"   扫描第 {i+1} 页成功 →: {detected_text[:50]}...")

    return {**state, "image_descriptions": descriptions, "status": "images_analyzed"}


def generate_story_node(state: StoryState) -> StoryState:
    """
    【不扩写模式】：只做去噪、纠错和微调通顺度，保留原汁原味
    """
    with timer("节点3 · 文本去噪与纯净微调（Ollama " + state["ollama_model"] + "）"):
        total = len(state["image_paths"])
        descs = "\n".join(state["image_descriptions"])
        theme = state["theme"] or "睡前故事"

        # 🚀 核心修改：明确限制 AI 不盲目编造和拉长篇幅，仅做纠错净化
        prompt = f"""你是优秀的儿童绘本文字编辑。
你的任务是对每页图片中 OCR 提取的原始文字线索进行“净化、纠错与微调”，使其更适合{state["age_group"]}儿童聆听。

【输入线索】
主线主题：{theme}
各页原始文字线索：
{descs}

【处理规则】
1. 严禁盲目扩写剧情或增加不必要的废话。必须最大程度保留原本文字的核心词汇和句式结构。
2. 纠正 OCR 识别中可能出现的错别字、不连贯的断句或乱码。如果句子读起来不通顺，请进行轻微润色使其口语化、平缓。
3. 如果某页的线索提示为“未识别出明显文字”，请结合上下文主题《{theme}》，为该页创作一句极其简练（20字以内）的剧情连接句。

请严格按以下格式逐页输出，每页一行，不要有任何其他解释或前后缀：
第1页：[净化后的文字内容]
第2页：[净化后的文字内容]
{"".join(f"第{i+1}页：[净化后的文字内容]" + chr(10) for i in range(2, total))}
最后页尾请自然以类似“宝贝，晚安”的安抚语结尾。"""

        raw = ollama_chat(state["ollama_model"], prompt, timeout=180)
        
        import re
        lines = []
        for i in range(1, total + 1):
            m = re.search(rf'第\s*{i}\s*页[：:]\s*(.+)', raw)
            if m:
                lines.append(m.group(1).strip())
        
        if len(lines) == total:
            texts = lines
        else:
            print(f"   ⚠ 整体解析失败，正在启动单页净化保底...")
            texts = []
            for i in range(total):
                p = f"请对这句话进行错别字修正和语顺微调，禁止添加新剧情和盲目扩写，直接输出结果：{state['image_descriptions'][i]}"
                t = ollama_chat(state["ollama_model"], p, timeout=60).strip()
                t = re.sub(r'^第\d+页[：:]\s*', '', t) 
                texts.append(t or f"翻到第{i+1}页啦。")
                print(f"   兜底第 {i+1} 页: {texts[-1]}")

        print(f"   故事文本净化成功，共 {len(texts)} 页")

    return {**state, "story_texts": texts, "status": "story_generated"}


def _detect_tts_engine() -> str:
    import socket
    try:
        socket.setdefaulttimeout(3)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("speech.platform.bing.com", 443))
        return "edge-tts"
    except Exception:
        pass
    try:
        import pyttsx3
        return "pyttsx3"
    except ImportError:
        pass
    return "silent"


async def _tts_edge(text: str, path: str, voice: str, speed: float):
    rate = f"{int((speed - 1) * 100):+d}%"
    for attempt in range(3):
        try:
            await edge_tts.Communicate(text, voice, rate=rate).save(path)
            return True
        except Exception:
            await asyncio.sleep((attempt + 1) * 2)
    return False


def _tts_pyttsx3(text: str, path: str, speed: float):
    import pyttsx3
    tmp_aiff = path.replace(".mp3", ".aiff")
    engine = pyttsx3.init()
    for voice in engine.getProperty("voices"):
        if any(n in voice.name for n in ["Tingting", "Sinji", "Chinese", "中文"]):
            engine.setProperty("voice", voice.id)
            break
    engine.setProperty("rate", int(200 * speed))
    engine.save_to_file(text, tmp_aiff)
    engine.runAndWait()
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", tmp_aiff, "-c:a", "libmp3lame", "-q:a", "4", path
    ])
    if os.path.exists(tmp_aiff):
        os.remove(tmp_aiff)


def _tts_silent(path: str, duration: float = 5.0):
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
        "-t", str(duration), "-c:a", "libmp3lame", path
    ])


def synthesize_audio_node(state: StoryState) -> StoryState:
    engine = _detect_tts_engine()
    with timer(f"节点4 · 语音合成（{engine}）"):
        audio_dir = Path(state["output_dir"]) / "audio"
        audio_dir.mkdir(exist_ok=True)

        if engine == "edge-tts":
            async def run_edge():
                paths = []
                for i, text in enumerate(state["story_texts"]):
                    p = str(audio_dir / f"page_{i+1:02d}.mp3")
                    ok = await _tts_edge(text, p, state["voice_name"], state["voice_speed"])
                    if not ok:
                        _tts_silent(p)
                    paths.append(p)
                return paths
            paths = asyncio.run(run_edge())
        elif engine == "pyttsx3":
            paths = []
            for i, text in enumerate(state["story_texts"]):
                p = str(audio_dir / f"page_{i+1:02d}.mp3")
                try:
                    _tts_pyttsx3(text, p, state["voice_speed"])
                except Exception:
                    _tts_silent(p)
                paths.append(p)
        else:
            paths = []
            for i in range(len(state["story_texts"])):
                p = str(audio_dir / f"page_{i+1:02d}.mp3")
                _tts_silent(p)
                paths.append(p)

        durations = []
        for p in paths:
            try:
                r = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", p],
                    capture_output=True, text=True, timeout=10
                )
                durations.append(float(json.loads(r.stdout)["streams"][0]["duration"]))
            except Exception:
                durations.append(5.0)

    return {**state, "audio_paths": paths, "audio_durations": durations, "status": "audio_synthesized"}


def render_frames_node(state: StoryState) -> StoryState:
    with timer("节点5 · 渲染字幕帧"):
        w, h = state["resolution"]
        frame_dir = Path(state["output_dir"]) / "frames"
        frame_dir.mkdir(exist_ok=True)
        frame_paths = []

        for i, (img_path, text) in enumerate(zip(state["image_paths"], state["story_texts"])):
            dst = str(frame_dir / f"frame_{i+1:02d}.jpg")
            draw_subtitle(img_path, dst, text, w, h)
            frame_paths.append(dst)
        print(f"   已离线绘制完 {len(frame_paths)} 张全真字幕图片")

    return {**state, "frame_paths": frame_paths, "status": "frames_rendered"}


def _videotoolbox_available() -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=5)
        return "h264_videotoolbox" in r.stdout
    except Exception:
        return False


def compose_video_node(state: StoryState) -> StoryState:
    use_hw = _videotoolbox_available()
    codec_label = "VideoToolbox 硬件加速" if use_hw else "libx264 ultrafast"
    with timer(f"节点6 · 视频合成（{codec_label}）"):
        video_dir = Path(state["output_dir"]) / "video"
        video_dir.mkdir(exist_ok=True)
        w, h = state["resolution"]

        with tempfile.TemporaryDirectory() as tmp:
            seg_paths = []
            for i, (frame, audio, dur_audio) in enumerate(zip(state["frame_paths"], state["audio_paths"], state["audio_durations"])):
                dur = state["page_duration"] if state["page_duration"] > 0 else dur_audio + 0.3
                seg = os.path.join(tmp, f"seg_{i:03d}.mp4")

                cmd = ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", frame]
                if audio and os.path.exists(audio):
                    cmd += ["-i", audio, "-c:a", "aac", "-b:a", "128k", "-shortest"]

                if use_hw:
                    cmd += ["-c:v", "h264_videotoolbox", "-b:v", "2000k", "-realtime", "false", "-t", str(dur), "-vf", f"scale={w}:{h},format=yuv420p", "-r", "24", seg]
                else:
                    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-t", str(dur), "-vf", f"scale={w}:{h},format=yuv420p", "-r", "24", seg]

                subprocess.run(cmd, check=True)
                seg_paths.append(seg)

            concat_file = os.path.join(tmp, "concat.txt")
            with open(concat_file, "w") as f:
                for seg in seg_paths:
                    f.write(f"file '{seg}'\n")

            output_path = str(video_dir / "story_video.mp4")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", concat_file, "-c", "copy", output_path], check=True)

        print(f"   视频无损生成完毕: {output_path}")

    return {**state, "video_path": output_path, "status": "video_composed"}


def save_output_node(state: StoryState) -> StoryState:
    with timer("节点7 · 保存清单"):
        story_data = {
            "title": state["story_texts"][0][:10] if state["story_texts"] else "睡前故事",
            "theme": state["theme"],
            "age_group": state["age_group"],
            "style": state["story_style"],
            "pages": [
                {
                    "page": i + 1,
                    "image": state["image_paths"][i],
                    "text":  state["story_texts"][i],
                    "audio": state["audio_paths"][i],
                    "duration": state["audio_durations"][i]
                } for i in range(len(state["image_paths"]))
            ],
            "video": state["video_path"],
        }
        json_path = str(Path(state["output_dir"]) / "story.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(story_data, f, ensure_ascii=False, indent=2)

    print(f"\n🌙 故事生成成功！\n🎬 视频文件 : {state['video_path']}\n📄 清单文件 : {json_path}\n")
    return {**state, "story_json_path": json_path, "status": "completed"}


# ══════════════════════════════════════════════════════════════════════════════
# 4. 组装图工作流
# ══════════════════════════════════════════════════════════════════════════════

def build_workflow():
    wf = StateGraph(StoryState)
    wf.add_node("init",            init_node)
    wf.add_node("analyze_images",  analyze_images_node)
    wf.add_node("generate_story",  generate_story_node)
    wf.add_node("synthesize_audio",synthesize_audio_node)
    wf.add_node("render_frames",   render_frames_node)
    wf.add_node("compose_video",   compose_video_node)
    wf.add_node("save_output",     save_output_node)

    wf.set_entry_point("init")
    wf.add_edge("init",             "analyze_images")
    wf.add_edge("analyze_images",   "generate_story")
    wf.add_edge("generate_story",   "synthesize_audio")
    wf.add_edge("synthesize_audio", "render_frames")
    wf.add_edge("render_frames",    "compose_video")
    wf.add_edge("compose_video",    "save_output")
    wf.add_edge("save_output",      END)
    return wf.compile()


def parse_args():
    p = argparse.ArgumentParser(description="每晚讲故事书 — 完美适配本地 OCR 扩展版")
    p.add_argument("--images",       default="./images/",           help="图片目录")
    p.add_argument("--output",       default="./stories/output/",   help="输出目录")
    p.add_argument("--theme",        default="绘本原有文字扩展",       help="全局基调主题")
    p.add_argument("--age",          default="3-6岁",                help="目标年龄段")
    p.add_argument("--style",        default="温柔安抚",              help="故事风格")
    p.add_argument("--voice",        default="zh-CN-XiaoxiaoNeural", help="TTS 发音人")
    p.add_argument("--speed",        type=float, default=0.85,       help="语速")
    p.add_argument("--resolution",   default="1280x720",             help="视频分辨率")
    p.add_argument("--page-duration",type=float, default=0.0,        help="单页时长固定(秒)，0=随音频")
    p.add_argument("--model",        default="qwen2.5:3b",           help="本地 Ollama 模型")
    return p.parse_args()


# 🌟 显式向外暴露给 LangGraph 服务的全局变量（对接 Studio 必备入口）
story_app = build_workflow()


def main():
    args = parse_args()
    p = Path(args.images)
    exts = {".jpg",".jpeg",".png",".webp"}
    images = sorted(str(f) for f in p.iterdir() if f.suffix.lower() in exts) if p.is_dir() else []
    
    if not images:
        print(f"❌ 错误：在 '{args.images}' 目录下未检测到任何有效绘本图片！")
        return

    try:
        w, h = map(int, args.resolution.lower().split("x"))
    except Exception:
        w, h = 1280, 720

    print(f"🏁 准备就绪：开始分析并提取 {len(images)} 张图片的文字内容...")

    state: StoryState = {
        "image_paths":       images,
        "raw_image_paths":   images, 
        "output_dir":        args.output,
        "theme":             args.theme,
        "age_group":         args.age,
        "story_style":       args.style,
        "voice_name":        args.voice,
        "voice_speed":       args.speed,
        "resolution":        (w, h),
        "page_duration":     args.page_duration,
        "ollama_model":      args.model,
        "image_descriptions":[],
        "story_texts":       [],
        "audio_paths":       [],
        "audio_durations":   [],
        "frame_paths":       [],
        "video_path":        "",
        "story_json_path":   "",
        "status":            "pending",
    }

    story_app.invoke(state)


if __name__ == "__main__":
    main()
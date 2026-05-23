import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import tempfile
import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import io
import json
import requests
import base64
from pathlib import Path
from typing import Optional

# --- 1. UI 基本准则配置 --- [cite: 21, 22, 23]
st.set_page_config(layout="wide", page_title="BLADE STUDIO")
MAIN_COLOR = "#2F54EB"
ERROR_COLOR = "#F5222D"
ACCENT_COLOR = "#91C1FF"
NEUTRAL_BG = "#EEF2F7"
TEXT_COLOR = "#1F2A44"
CLASS_COLOR_MAP = {
    "前缘腐蚀": (245, 34, 45),   # 红色
    "裂纹": (250, 140, 22),      # 橙色
    "默认": (82, 196, 26)        # 绿色
}
LABEL_ALIAS_MAP = {
    "前缘腐蚀": "LeadingEdgeCorrosion",
    "裂纹": "Crack",
}
VIDEO_URL = (
    "https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/"
    "hf_20260328_083109_283f3553-e28f-428b-a723-d639c617eb2b.mp4"
)
CINE_BLACK = "#000000"
CINE_MUTED = "#6F6F6F"
CINE_WHITE = "#FFFFFF"

PAGE_HERO = {
    "检测": {
        "title": "穿透噪声，<em>看见</em>每一处缺陷",
        "desc": "上传模型与叶片图像，完成智能检测、可视化标注与结果导出。",
    },
    "历史数据": {
        "title": "每一次检测，<em>皆有迹可循</em>",
        "desc": "按类型、置信度与时间维度筛选历史记录，追溯检测全过程。",
    },
    "AI 分析": {
        "title": "让数据开口，<em>辅助决策</em>",
        "desc": "基于检测结果图与结构化缺陷信息，进行对话式智能分析。",
    },
}
BG_IMAGE_PATHS = [
    Path(__file__).parent / "site_bg.jpg",
    Path(__file__).parent / "assets" / "site_bg.jpg",
]
LOGO_CANDIDATE_PATHS = [
    Path(__file__).parent / "logo_blade_detection_studio.png",
    Path(__file__).parent / "assets" / "logo_blade_detection_studio.png",
    Path(__file__).parent / "assets" / "logo.png",
    Path(__file__).parent / "logo.png",
]


def _pick_box_color(label: str):
    if "前缘腐蚀" in label:
        return CLASS_COLOR_MAP["前缘腐蚀"]
    if "裂纹" in label:
        return CLASS_COLOR_MAP["裂纹"]
    return CLASS_COLOR_MAP["默认"]


def _alias_label_for_box(label: str, cls_idx: int):
    if label in LABEL_ALIAS_MAP:
        return LABEL_ALIAS_MAP[label]
    try:
        label.encode("ascii")
        return label
    except UnicodeEncodeError:
        return f"class_{cls_idx}"


def draw_annotated_image(image_pil: Image.Image, result, model_names):
    image_rgb = np.array(image_pil.convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    defect_items = []

    boxes = result.boxes if result.boxes is not None else []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model_names[cls]
        box_label = _alias_label_for_box(label, cls)
        color_rgb = _pick_box_color(label)
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        # 绘制检测框和标签，颜色由缺陷类型决定
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color_bgr, 2)
        text = f"{box_label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(image_bgr, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color_bgr, -1)
        cv2.putText(image_bgr, text, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        defect_items.append({"label": label, "conf": conf})

    annotated_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, defect_items


def image_to_png_bytes(image_rgb: np.ndarray):
    image_pil = Image.fromarray(image_rgb.astype(np.uint8))
    buf = io.BytesIO()
    image_pil.save(buf, format="PNG")
    return buf.getvalue()


def build_report_text(record: dict):
    lines = [
        "BLADE STUDIO 检测报告",
        f"时间: {record['time']}",
        f"图片: {record['image_name']}",
        f"模型: {record['model_name']}",
        f"缺陷总数: {record['defect_count']}",
        ""
    ]
    if not record["defects"]:
        lines.append("未发现缺陷。")
    else:
        for idx, item in enumerate(record["defects"], start=1):
            lines.append(f"缺陷 {idx}: {item['label']} (置信度 {item['conf'] * 100:.1f}%)")
    return "\n".join(lines)


def load_bg_data_uri():
    for p in BG_IMAGE_PATHS:
        if p.exists():
            suffix = p.suffix.lower()
            mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
            encoded = base64.b64encode(p.read_bytes()).decode("utf-8")
            return f"data:{mime};base64,{encoded}"
    return None


def render_cinematic_hero(page_name: str, logo_uri: Optional[str]):
    hero = PAGE_HERO.get(page_name, PAGE_HERO["检测"])
    logo_html = f'<img src="{logo_uri}" class="cine-logo" alt="logo"/>' if logo_uri else ""
    st.markdown(
        f"""
        <div class="cine-hero">
            <div class="cine-nav">
                <div class="cine-nav-left">{logo_html}<span class="cine-brand">BLADE STUDIO<sup>®</sup></span></div>
                <span class="cine-nav-meta">风电叶片缺陷检测分析系统</span>
            </div>
            <div class="cine-hero-body">
                <h1 class="cine-headline animate-fade-rise">{hero["title"]}</h1>
                <p class="cine-desc animate-fade-rise-delay">{hero["desc"]}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_cinematic_video():
    st.markdown(
        f"""
        <div class="cine-video-wrap">
            <video id="cine-bg-video" class="cine-video" autoplay muted playsinline loop>
                <source src="{VIDEO_URL}" type="video/mp4"/>
            </video>
            <div class="cine-video-gradient"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    components.html(
        """
        <script>
        (function () {
            const doc = window.parent.document;
            const video = doc.getElementById("cine-bg-video");
            if (!video || video.dataset.cineBound === "1") return;
            video.dataset.cineBound = "1";
            video.removeAttribute("loop");
            const FADE = 0.5;
            const tick = () => {
                const d = video.duration;
                if (d && isFinite(d)) {
                    let op = 1;
                    const t = video.currentTime;
                    if (t < FADE) op = t / FADE;
                    else if (d - t < FADE) op = (d - t) / FADE;
                    video.style.opacity = Math.max(0, Math.min(1, op));
                }
                requestAnimationFrame(tick);
            };
            video.addEventListener("ended", () => {
                video.style.opacity = "0";
                setTimeout(() => { video.currentTime = 0; video.play(); }, 100);
            });
            video.addEventListener("canplay", () => video.play());
            requestAnimationFrame(tick);
        })();
        </script>
        """,
        height=0,
    )


def load_logo_data_uri():
    target_bg = np.array([222, 229, 240], dtype=np.uint8)  # 侧边栏浅灰蓝底色
    for p in LOGO_CANDIDATE_PATHS:
        if p.exists():
            try:
                img = Image.open(p).convert("RGBA")
                arr = np.array(img)
                rgb = arr[:, :, :3]

                # 将接近白色的背景替换为侧边栏颜色，保留图标主体
                white_mask = (rgb[:, :, 0] > 225) & (rgb[:, :, 1] > 225) & (rgb[:, :, 2] > 225)
                arr[white_mask, :3] = target_bg

                out = Image.fromarray(arr, mode="RGBA")
                buf = io.BytesIO()
                out.save(buf, format="PNG")
                encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
                return f"data:image/png;base64,{encoded}"
            except Exception:
                encoded = base64.b64encode(p.read_bytes()).decode("utf-8")
                return f"data:image/png;base64,{encoded}"
    return None


def generate_ai_reply(user_text: str, record: dict):
    if not record:
        return "当前没有可分析的检测结果。请先在「检测」页面完成一次检测。"

    defect_count = record.get("defect_count", 0)
    max_conf = record.get("max_conf", 0.0)
    types = record.get("defect_types", [])
    type_text = "、".join(types) if types else "无"
    question = user_text.lower()

    if "总结" in user_text or "概览" in user_text:
        return (
            f"本次检测共识别到 **{defect_count}** 处缺陷，缺陷类型为：**{type_text}**，"
            f"最高置信度 **{max_conf * 100:.1f}%**。"
        )

    if "建议" in user_text or "处理" in user_text or "维修" in user_text:
        if defect_count == 0:
            return "未发现缺陷，建议保持常规巡检周期，并持续记录环境变化。"
        return (
            "建议按以下优先级处理：\n"
            f"1) 优先复核高置信度区域（最高 {max_conf * 100:.1f}%）；\n"
            "2) 对同类型缺陷进行集中评估；\n"
            "3) 结合现场工况安排复检与维护。"
        )

    if "风险" in user_text or "严重" in user_text:
        if defect_count == 0:
            return "当前结果显示风险较低，但建议持续监测关键部位。"
        risk_level = "高" if max_conf >= 0.8 else ("中" if max_conf >= 0.5 else "低")
        return f"基于当前检测结果，建议风险等级评估为 **{risk_level}**。请结合人工复核确认最终结论。"

    if "置信度" in user_text or "confidence" in question:
        return f"当前最高置信度为 **{max_conf * 100:.1f}%**，建议优先核查该区域。"

    return (
        "我已基于当前检测结果建立上下文。你可以继续问我：\n"
        "- 帮我总结本次缺陷\n"
        "- 给出维修建议\n"
        "- 评估当前风险等级\n"
        "- 按置信度排序说明重点区域"
    )


def call_llm_api(user_text: str, record: dict, chat_history: list, llm_cfg: dict):
    if not llm_cfg.get("api_key"):
        return False, "未配置 API Key。请在侧边栏填写 Key 后重试。"

    base_url = llm_cfg.get("base_url", "https://api.deepseek.com/v1").rstrip("/")
    model = llm_cfg.get("model", "deepseek-chat")
    endpoint = f"{base_url}/chat/completions"

    record_summary = (
        f"时间: {record.get('time', '-')}\n"
        f"图片: {record.get('image_name', '-')}\n"
        f"模型: {record.get('model_name', '-')}\n"
        f"缺陷总数: {record.get('defect_count', 0)}\n"
        f"缺陷类型: {', '.join(record.get('defect_types', [])) or '无'}\n"
        f"最高置信度: {record.get('max_conf', 0.0) * 100:.1f}%"
    )
    defect_lines = []
    for i, d in enumerate(record.get("defects", []), start=1):
        defect_lines.append(f"{i}. {d.get('label', 'unknown')} ({d.get('conf', 0.0) * 100:.1f}%)")
    defect_detail = "\n".join(defect_lines) if defect_lines else "无缺陷"

    system_prompt = (
        "你是风电叶片缺陷检测助手。回答必须使用中文简体，简洁、专业、可执行。"
        "根据给定检测结果进行分析，不要编造不存在的数据。"
        "\n\n[检测概览]\n"
        f"{record_summary}\n\n[缺陷明细]\n{defect_detail}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {llm_cfg['api_key']}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=90)
        if resp.status_code != 200:
            return False, f"API 调用失败（HTTP {resp.status_code}）：{resp.text[:300]}"
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return True, content
    except Exception as e:
        return False, f"API 调用异常：{e}"

logo_data_uri = load_logo_data_uri()
bg_data_uri = load_bg_data_uri()
bg_css = ""
if bg_data_uri:
    bg_css = f"""
    .stApp {{
        background:
            linear-gradient(180deg, rgba(255,255,255,0.88) 0%, rgba(255,255,255,0.72) 45%, rgba(255,255,255,0.88) 100%),
            url("{bg_data_uri}") center center / cover no-repeat fixed !important;
    }}
    """
else:
    bg_css = f"""
    .stApp {{
        background: linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(238,242,247,0.85) 100%) !important;
    }}
    """

# 自定义 CSS — Cinematic Hero 风格融入 BLADE STUDIO
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600&display=swap');
    {bg_css}
    @keyframes fadeSlideUp {{
        from {{ opacity: 0; transform: translateY(18px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to   {{ opacity: 1; }}
    }}
    @keyframes slideInLeft {{
        from {{ opacity: 0; transform: translateX(-12px); }}
        to   {{ opacity: 1; transform: translateX(0); }}
    }}
    @keyframes navPulse {{
        0%, 100% {{ box-shadow: 0 0 0 0 rgba(22, 119, 255, 0); }}
        50%      {{ box-shadow: 0 0 0 4px rgba(22, 119, 255, 0.12); }}
    }}
    @keyframes shimmer {{
        0%   {{ background-position: -200% 0; }}
        100% {{ background-position: 200% 0; }}
    }}

    @keyframes fade-rise {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .animate-fade-rise {{ animation: fade-rise 0.8s ease-out both; }}
    .animate-fade-rise-delay {{ animation: fade-rise 0.8s ease-out 0.2s both; }}
    .animate-fade-rise-delay-2 {{ animation: fade-rise 0.8s ease-out 0.4s both; }}

    html, body, [class*="css"] {{
        font-family: "Inter", "PingFang SC", "Microsoft YaHei", sans-serif !important;
        scroll-behavior: smooth;
    }}
    h1, h2, h3, .cine-brand, .cine-headline {{
        font-family: "Instrument Serif", "PingFang SC", serif !important;
    }}
    .stApp {{
        color: {CINE_BLACK};
        position: relative;
    }}
    /* 电影感视频背景层 */
    .cine-video-wrap {{
        position: fixed;
        top: 300px;
        left: 0; right: 0; bottom: 0;
        z-index: 0;
        overflow: hidden;
        pointer-events: none;
    }}
    .cine-video {{
        width: 100%; height: 100%;
        object-fit: cover;
        opacity: 0;
    }}
    .cine-video-gradient {{
        position: absolute; inset: 0;
        background: linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.2) 40%, rgba(255,255,255,0.92) 100%);
    }}
    [data-testid="stAppViewContainer"] {{
        position: relative;
        z-index: 1;
    }}
    section[data-testid="stSidebar"], div[data-testid="stSidebar"] {{
        background: rgba(255, 255, 255, 0.72) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(0,0,0,0.06) !important;
        z-index: 2;
    }}
    [data-testid="stAppViewContainer"] .main .block-container {{
        background: transparent;
        max-width: 1200px;
        padding-top: 0.25rem;
        padding-bottom: 2rem;
    }}
    /* Cinematic Hero */
    .cine-hero {{
        margin-bottom: 28px;
        padding: 0 4px;
    }}
    .cine-nav {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0 20px;
        border-bottom: 1px solid rgba(0,0,0,0.06);
        margin-bottom: 28px;
    }}
    .cine-nav-left {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .cine-logo {{
        height: 52px;
        width: auto;
        border-radius: 6px;
    }}
    .cine-brand {{
        font-size: 1.75rem;
        letter-spacing: -0.02em;
        color: {CINE_BLACK};
        font-weight: 400;
    }}
    .cine-brand sup {{ font-size: 0.55em; }}
    .cine-nav-meta {{
        font-size: 0.82rem;
        color: {CINE_MUTED};
        font-weight: 500;
    }}
    .cine-hero-body {{
        text-align: center;
        padding: 12px 12px 8px;
    }}
    .cine-headline {{
        font-size: clamp(2rem, 5vw, 3.5rem);
        font-weight: 400;
        line-height: 0.95;
        letter-spacing: -2.46px;
        color: {CINE_BLACK};
        margin: 0;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
    }}
    .cine-headline em {{
        font-style: italic;
        color: {CINE_MUTED};
    }}
    .cine-desc {{
        font-size: 1rem;
        line-height: 1.65;
        color: {CINE_MUTED};
        max-width: 640px;
        margin: 20px auto 0;
    }}
    /* 主内容区切换动画容器 */
    .page-shell {{
        animation: fadeSlideUp 0.42s cubic-bezier(0.22, 1, 0.36, 1) both;
    }}
    .page-shell .module-box {{
        animation: fadeSlideUp 0.5s cubic-bezier(0.22, 1, 0.36, 1) both;
    }}
    .page-shell .module-box:nth-child(1) {{ animation-delay: 0.04s; }}
    .page-shell .module-box:nth-child(2) {{ animation-delay: 0.08s; }}
    .page-shell .module-box:nth-child(3) {{ animation-delay: 0.12s; }}
    .page-title {{
        animation: slideInLeft 0.38s cubic-bezier(0.22, 1, 0.36, 1) both;
        margin-bottom: 0.25rem;
    }}
    .page-caption {{
        animation: fadeIn 0.45s ease both;
        animation-delay: 0.06s;
        color: #5B6B88;
        font-size: 13px;
        margin-bottom: 1rem;
    }}
    header[data-testid="stHeader"] {{
        display: none !important;
        height: 0 !important;
    }}
    div[data-testid="stToolbar"] {{
        display: none !important;
    }}
    [data-testid="stAppViewContainer"] > .main {{
        padding-top: 0 !important;
    }}
    [data-testid="stAppViewContainer"] .main .block-container {{
        max-width: 1200px;
        padding-top: 0.25rem;
        padding-bottom: 2rem;
    }}
    [data-testid="stAppViewContainer"] .main .stButton>button {{
        background-color: {CINE_BLACK};
        color: {CINE_WHITE};
        border-radius: 999px;
        height: 44px;
        width: 100%;
        border: none;
        font-weight: 600;
        font-family: "Inter", sans-serif !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    [data-testid="stAppViewContainer"] .main .stButton>button:hover {{
        transform: scale(1.03);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }}
    section[data-testid="stSidebar"] .stButton>button {{
        width: 100%;
        min-height: 46px;
        border-radius: 10px;
        font-size: 15px;
        font-weight: 600;
        border: 1px solid transparent;
        background: transparent;
        color: #4B5568 !important;
        box-shadow: none !important;
        text-align: left;
        padding-left: 16px;
        transition: all 0.22s cubic-bezier(0.22, 1, 0.36, 1);
        position: relative;
        overflow: hidden;
    }}
    section[data-testid="stSidebar"] .stButton>button * {{
        color: inherit !important;
        transition: color 0.2s ease;
    }}
    section[data-testid="stSidebar"] .stButton>button:hover {{
        border-color: #D0DAEA;
        background: rgba(255, 255, 255, 0.55);
        color: #1F2A44 !important;
        transform: translateX(3px);
    }}
    section[data-testid="stSidebar"] .stButton>button[kind="primary"] {{
        background: rgba(255,255,255,0.95);
        border-color: rgba(0,0,0,0.08);
        color: {CINE_BLACK} !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06) !important;
    }}
    section[data-testid="stSidebar"] .stButton>button[kind="primary"]::before {{
        content: "";
        position: absolute;
        left: 0; top: 8px; bottom: 8px;
        width: 3px;
        border-radius: 0 3px 3px 0;
        background: {CINE_BLACK};
    }}
    .sidebar-brand {{
        font-family: "Instrument Serif", serif !important;
        font-size: 1.4rem;
        color: {CINE_BLACK};
        margin-bottom: 4px;
    }}
    .sidebar-sub {{
        color: {CINE_MUTED};
        font-size: 12px;
        margin-bottom: 16px;
    }}
    .sidebar-nav-wrap {{
        display: grid;
        gap: 10px;
        margin-top: 8px;
        margin-bottom: 8px;
    }}
    .sidebar-divider {{
        border-top: 1px dashed #8FA7D8;
        margin: 14px 0;
    }}
    .sidebar-chip {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 11px;
        background: rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.08);
        color: {CINE_MUTED};
        font-weight: 600;
        letter-spacing: 0.02em;
    }}
    .module-box {{
        padding: 22px 24px;
        border-radius: 16px;
        border: 1px solid rgba(0,0,0,0.06);
        margin-bottom: 16px;
        background: rgba(255, 255, 255, 0.78);
        backdrop-filter: blur(14px);
        box-shadow: 0 4px 24px rgba(0,0,0,0.04);
        transition: box-shadow 0.25s ease, transform 0.25s ease;
    }}
    .module-box:hover {{
        box-shadow: 0 6px 20px rgba(31, 42, 68, 0.09);
        transform: translateY(-1px);
    }}
    .module-config {{ border-left: 4px solid #2F54EB; }}
    .module-analysis {{ border-left: 4px solid #52C41A; }}
    .module-upload {{ border-left: 4px solid #FAAD14; }}
    .module-divider {{ margin: 4px 0 16px 0; border-top: 1px solid #D8E0ED; }}
    .glass-panel {{
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(200, 212, 232, 0.75);
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 2px 12px rgba(31, 42, 68, 0.06);
        backdrop-filter: blur(10px);
        animation: fadeSlideUp 0.45s cubic-bezier(0.22, 1, 0.36, 1) both;
    }}
    .model-file-hint {{ font-size: 13px; margin-bottom: 8px; }}
    .model-file-hint.unselected {{ color: #8C8C8C; }}
    .model-file-hint.selected {{ color: #2F54EB; font-weight: 600; }}
    .status-chip {{
        display: inline-block;
        color: white;
        padding: 4px 12px;
        border-radius: 999px;
        margin-right: 8px;
        font-size: 13px;
        font-weight: 600;
        transition: transform 0.2s ease;
    }}
    .status-chip:hover {{ transform: scale(1.04); }}
    div[data-testid="stChatMessage"] {{
        animation: fadeSlideUp 0.35s cubic-bezier(0.22, 1, 0.36, 1) both;
    }}
    [data-testid="stExpander"] {{
        animation: fadeSlideUp 0.38s cubic-bezier(0.22, 1, 0.36, 1) both;
        border-radius: 10px !important;
        border: 1px solid #D8E0ED !important;
        background: rgba(255,255,255,0.6) !important;
        transition: box-shadow 0.2s ease;
    }}
    [data-testid="stExpander"]:hover {{
        box-shadow: 0 4px 14px rgba(31, 42, 68, 0.07);
    }}
    [data-testid="stAppViewContainer"] .main .stButton>button {{
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }}
    [data-testid="stAppViewContainer"] .main .stButton>button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(47, 84, 235, 0.25);
    }}
    [data-testid="stAppViewContainer"] .main .stButton>button:active {{
        transform: translateY(0);
    }}
    .status-pending {{ background-color: #FAAD14; }}
    .status-detected {{ background-color: #52C41A; }}
    .status-defect {{ background-color: #F5222D; }}
    div[data-testid="stTextInput"] input {{
        color: #8C8C8C;
    }}
    div[data-testid="stTextInput"] input:not(:placeholder-shown) {{
        color: #2F54EB;
        font-weight: 600;
    }}
    div[data-testid="stBottomBlockContainer"],
    div[data-testid="stChatFloatingInputContainer"],
    div[data-testid="stChatInputContainer"],
    div[data-testid="stChatInputContainer"] > div,
    div[data-testid="stChatInput"] > div,
    div[data-testid="stChatInput"] textarea {{
        background: rgba(255,255,255,0.75) !important;
        backdrop-filter: blur(12px);
    }}
    div[data-testid="stChatInput"] > div {{
        border: 1px solid rgba(0,0,0,0.08) !important;
        border-radius: 999px !important;
    }}
    div[data-testid="stChatInput"] button {{
        background: {CINE_BLACK} !important;
        color: {CINE_WHITE} !important;
        border-radius: 999px !important;
    }}
    .app-footer {{
        min-height: 80px;
        margin-top: 18px;
        border-radius: 12px;
        background: rgba(145, 193, 255, 0.15);
        border: 1px solid rgba(145, 193, 255, 0.45);
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 20px;
        color: #304670;
        font-size: 13px;
        font-weight: 600;
    }}
    </style>
""", unsafe_allow_html=True)

# 会在侧边栏中使用，需先初始化
if "llm_cfg" not in st.session_state:
    st.session_state.llm_cfg = {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "api_key": ""
    }

# --- 2. 侧边栏导航 --- 
with st.sidebar:
    st.markdown('<div class="sidebar-brand">BLADE STUDIO</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">风电叶片缺陷检测分析系统</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-chip">系统在线</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    if "page" not in st.session_state:
        st.session_state.page = "检测"

    def _switch_page(target_page: str):
        if st.session_state.page != target_page:
            st.session_state.prev_page = st.session_state.page
            st.session_state.page = target_page
            st.rerun()

    st.markdown('<div class="sidebar-nav-wrap">', unsafe_allow_html=True)
    if st.button("🔍  检测", key="nav_detect", use_container_width=True, type="primary" if st.session_state.page == "检测" else "secondary"):
        _switch_page("检测")
    if st.button("📋  历史数据", key="nav_history", use_container_width=True, type="primary" if st.session_state.page == "历史数据" else "secondary"):
        _switch_page("历史数据")
    if st.button("🤖  AI 分析", key="nav_ai", use_container_width=True, type="primary" if st.session_state.page == "AI 分析" else "secondary"):
        _switch_page("AI 分析")
    st.markdown('</div>', unsafe_allow_html=True)
    page = st.session_state.page
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.caption("LLM API 设置")
    st.session_state.llm_cfg["base_url"] = st.text_input(
        "Base URL",
        value=st.session_state.llm_cfg["base_url"],
        key="llm_base_url_input"
    )
    st.session_state.llm_cfg["model"] = st.text_input(
        "Model",
        value=st.session_state.llm_cfg["model"],
        key="llm_model_input"
    )
    st.session_state.llm_cfg["api_key"] = st.text_input(
        "API Key",
        value=st.session_state.llm_cfg["api_key"],
        type="password",
        key="llm_api_key_input"
    )

inject_cinematic_video()
render_cinematic_hero(page, logo_data_uri)

if "stats" not in st.session_state:
    st.session_state.stats = {"pending": 0, "detected": 0, "defects": 0}
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = []
if "prev_page" not in st.session_state:
    st.session_state.prev_page = "检测"

st.markdown('<div class="page-shell">', unsafe_allow_html=True)

if page == "检测":
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="module-box module-config">', unsafe_allow_html=True)
        st.subheader("模型与推理配置")
        model_source = st.radio("模型来源", ["本机选择 .pt 文件", "手动输入路径"], horizontal=True)

        model_path = ""
        if model_source == "本机选择 .pt 文件":
            uploaded_model_file = st.file_uploader("选择本机 .pt 模型文件", type=["pt"])
            if uploaded_model_file:
                if st.session_state.get("model_file_name") != uploaded_model_file.name:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                        tmp.write(uploaded_model_file.getbuffer())
                        st.session_state.model_temp_path = tmp.name
                        st.session_state.model_file_name = uploaded_model_file.name
                model_path = st.session_state.get("model_temp_path", "")
                st.markdown(
                    f'<div class="model-file-hint selected">📌 已选中模型：{uploaded_model_file.name}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown('<div class="model-file-hint unselected">📁 未选择 .pt 文件</div>', unsafe_allow_html=True)
        else:
            model_path = st.text_input("📁 输入 .pt 文件路径", placeholder="未选择 .pt 文件")
            if model_path:
                st.markdown(f'<div class="model-file-hint selected">📌 已选中模型：{model_path}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="model-file-hint unselected">📁 未选择 .pt 文件</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="module-box module-analysis">', unsafe_allow_html=True)
        st.subheader("结果分析")
        stats = st.session_state.stats
        st.markdown(
            f'<span class="status-chip status-pending">待检测: {stats["pending"]}</span>'
            f'<span class="status-chip status-detected">已检测: {stats["detected"]}</span>'
            f'<span class="status-chip status-defect">发现缺陷: {stats["defects"]}</span>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 4. 图片导入与执行 --- [cite: 15, 17]
    st.markdown('<div class="module-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="module-box module-upload">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("选择本机图片文件（可点击 Browse files）", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        st.markdown(f'<div class="model-file-hint selected">🖼️ 当前图片：{uploaded_file.name}</div>', unsafe_allow_html=True)
    st.session_state.stats["pending"] = 1 if uploaded_file else 0

    if st.button("执行检测"):
        if not uploaded_file or not model_path:
            st.error(
                "❌ 操作失败：缺少必要输入。\n\n"
                "原因：未上传图片或未指定模型。\n\n"
                "解决方法：\n"
                "1) 先选择本机图片文件；\n"
                "2) 再选择 .pt 模型（或输入有效模型路径）；\n"
                "3) 然后重新点击“执行检测”。"
            )
            st.toast("检测未执行：请先补全输入项", icon="⚠️")
        else:
            try:
                # 推理逻辑
                model = YOLO(model_path)
                img = Image.open(uploaded_file).convert("RGB")
                results = model(img)
            except Exception as e:
                st.error(
                    "❌ 模型加载或推理失败。\n\n"
                    f"原因：{e}\n\n"
                    "解决方法：\n"
                    "1) 确认模型文件与当前环境兼容；\n"
                    "2) 若提示缺少模块（如 aud_yolo），请补齐依赖源码；\n"
                    "3) 重新选择可用模型后再执行。"
                )
                st.toast("检测失败，请检查模型兼容性", icon="🚫")
                st.stop()

            annotated_img, defect_items = draw_annotated_image(img, results[0], model.names)
            max_conf = max([d["conf"] for d in defect_items], default=0.0)
            defect_types = sorted(list({d["label"] for d in defect_items}))
            record = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_name": uploaded_file.name,
                "model_name": st.session_state.get("model_file_name", model_path),
                "defect_count": len(defect_items),
                "defects": defect_items,
                "max_conf": max_conf,
                "defect_types": defect_types,
                "annotated_png": image_to_png_bytes(annotated_img)
            }

            st.session_state.stats["pending"] = 0
            st.session_state.stats["detected"] += 1
            st.session_state.stats["defects"] = len(defect_items)
            st.session_state.history.insert(0, record)
            report_record = {k: v for k, v in record.items() if k != "annotated_png"}
            st.session_state.last_result = {
                "record": record,
                "annotated_image": annotated_img,
                "report_text": build_report_text(record),
                "report_json": json.dumps(report_record, ensure_ascii=False, indent=2),
            }
            st.session_state.ai_messages = []
            st.success(f"✅ 检测完成：发现 {len(defect_items)} 处缺陷。")
            st.toast("检测完成，结果与历史记录已更新", icon="✅")

    # 结果展示与导出（保留最近一次结果，便于二次查看）
    if st.session_state.last_result:
        last_result = st.session_state.last_result
        result_col1, result_col2 = st.columns([2, 1])
        with result_col1:
            st.write("### 检测结果图")
            base_img = last_result["annotated_image"]
            ctrl_col1, ctrl_col2 = st.columns([1, 1])
            zoom_level = ctrl_col1.selectbox("缩放比例", ["100%", "150%", "200%"], index=0)
            fullscreen_mode = ctrl_col2.toggle("全屏查看", value=False)

            zoom_ratio = {"100%": 1.0, "150%": 1.5, "200%": 2.0}[zoom_level]
            show_img = base_img.copy()
            if zoom_ratio > 1.0:
                show_img = cv2.resize(show_img, dsize=None, fx=zoom_ratio, fy=zoom_ratio, interpolation=cv2.INTER_LINEAR)

            st.image(show_img, use_container_width=fullscreen_mode)
            st.markdown(
                '<span class="status-chip status-defect">前缘腐蚀</span>'
                '<span class="status-chip status-pending">裂纹</span>'
                '<span class="status-chip status-detected">其他缺陷</span>',
                unsafe_allow_html=True
            )

        with result_col2:
            st.write("### 检测摘要与分析结果")
            current = last_result["record"]
            st.info(f"检测到 {current['defect_count']} 处缺陷")
            for i, item in enumerate(current["defects"]):
                st.markdown(f"缺陷 {i + 1}：**{item['label']}**")
                p_col1, p_col2 = st.columns([4, 1])
                p_col1.progress(item["conf"])
                p_col2.markdown(f"`{item['conf'] * 100:.1f}%`")

            st.write("### 导出功能")
            report_col1, report_col2 = st.columns(2)
            report_col1.download_button(
                "导出报告(.txt)",
                data=last_result["report_text"],
                file_name=f"report_{current['time'].replace(':', '').replace(' ', '_')}.txt",
                mime="text/plain"
            )
            report_col2.download_button(
                "导出明细(.json)",
                data=last_result["report_json"],
                file_name=f"report_{current['time'].replace(':', '').replace(' ', '_')}.json",
                mime="application/json"
            )

            st.download_button(
                "保存结果图(.png)",
                data=image_to_png_bytes(last_result["annotated_image"]),
                file_name=f"result_{current['time'].replace(':', '').replace(' ', '_')}.png",
                mime="image/png"
            )
    st.markdown('</div>', unsafe_allow_html=True)
elif page == "历史数据":
    st.markdown('<div class="glass-panel animate-fade-rise-delay-2">', unsafe_allow_html=True)
    if not st.session_state.history:
        st.info("暂无历史记录。完成一次检测后会自动出现在这里。")
    else:
        st.caption(f"累计记录：{len(st.session_state.history)} 条")
        all_types = sorted({t for item in st.session_state.history for t in item.get("defect_types", [])})
        f_col1, f_col2 = st.columns(2)
        filter_type = f_col1.selectbox("按缺陷类型筛选", ["全部"] + all_types)
        min_conf = f_col2.slider("最小置信度", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

        s_col1, s_col2 = st.columns(2)
        sort_by = s_col1.selectbox("排序字段", ["时间", "缺陷数量", "最高置信度"])
        sort_order = s_col2.radio("排序方向", ["降序", "升序"], horizontal=True)
        reverse_sort = sort_order == "降序"

        filtered_history = []
        for item in st.session_state.history:
            type_ok = filter_type == "全部" or filter_type in item.get("defect_types", [])
            conf_ok = item.get("max_conf", 0.0) >= min_conf
            if type_ok and conf_ok:
                filtered_history.append(item)

        if sort_by == "时间":
            filtered_history = sorted(filtered_history, key=lambda x: x["time"], reverse=reverse_sort)
        elif sort_by == "缺陷数量":
            filtered_history = sorted(filtered_history, key=lambda x: x["defect_count"], reverse=reverse_sort)
        else:
            filtered_history = sorted(filtered_history, key=lambda x: x.get("max_conf", 0.0), reverse=reverse_sort)

        st.caption(f"筛选结果：{len(filtered_history)} 条")
        if not filtered_history:
            st.warning("当前筛选条件下没有匹配记录，请放宽筛选范围。")
        for idx, item in enumerate(filtered_history, start=1):
            with st.expander(f"记录 {idx} | {item['time']} | {item['image_name']} | 缺陷 {item['defect_count']} 处"):
                st.write(f"模型：{item['model_name']}")
                st.write(f"最高置信度：{item.get('max_conf', 0.0) * 100:.1f}%")
                if not item["defects"]:
                    st.success("本次未发现缺陷")
                else:
                    for i, defect in enumerate(item["defects"], start=1):
                        st.markdown(f"缺陷 {i}：**{defect['label']}**")
                        c1, c2 = st.columns([4, 1])
                        c1.progress(defect["conf"])
                        c2.markdown(f"`{defect['conf'] * 100:.1f}%`")
    st.markdown('</div>', unsafe_allow_html=True)
elif page == "AI 分析":

    if st.session_state.last_result:
        selected_record = st.session_state.last_result["record"]
    elif st.session_state.history:
        selected_record = st.session_state.history[0]
    else:
        selected_record = None

    if not selected_record:
        st.warning("暂无可分析结果。请先在「检测」页面完成一次检测。")
    else:
        image_data = selected_record.get("annotated_png")
        with st.container(border=True):
            st.markdown("#### 检测结果图")
            if image_data:
                st.image(image_data, use_container_width=True)
            else:
                st.info("该条历史记录缺少结果图（旧记录）。请重新执行一次检测生成可视化图像。")
            st.caption(
                f"时间：{selected_record['time']} | 图片：{selected_record['image_name']} | "
                f"缺陷数：{selected_record['defect_count']}"
            )

        chat_toolbar_col1, chat_toolbar_col2 = st.columns([3, 1])
        chat_toolbar_col1.markdown("#### LLM 对话")
        if chat_toolbar_col2.button("清空对话", use_container_width=True):
            st.session_state.ai_messages = []
            st.rerun()

        for msg in st.session_state.ai_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_prompt = st.chat_input("输入问题，例如：请总结本次缺陷并给出处理建议")
        if user_prompt:
            st.session_state.ai_messages.append({"role": "user", "content": user_prompt})
            ok, reply = call_llm_api(
                user_prompt,
                selected_record,
                st.session_state.ai_messages,
                st.session_state.llm_cfg
            )
            if not ok:
                # API 不可用时回退本地规则回答，保证可用性
                fallback = generate_ai_reply(user_prompt, selected_record)
                reply = f"{reply}\n\n---\n已切换本地分析回复：\n{fallback}"
            st.session_state.ai_messages.append({"role": "assistant", "content": reply})
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

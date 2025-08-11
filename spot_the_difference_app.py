import os
import io
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
from PIL import Image

import streamlit as st

# SAM
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# =========================
# 설정
# =========================
SAM_CHECKPOINT = "./sam_vit_h_4b8939.pth"  # <- 본인 경로로 변경
SAM_MODEL_TYPE = "vit_h"                 # vit_h / vit_l / vit_b

# 자동 마스크 파라미터(필요시 조절)
MASK_GEN_KW = dict(
    points_per_side=32,            # 더 촘촘히 (느려짐)
    pred_iou_thresh=0.88,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=256       # 너무 작은 마스크 제거
)


# =========================
# 유틸
# =========================
def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def bgr_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def ensure_three_channels(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        return np.stack([mask]*3, axis=2)
    return mask

def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-9)

def nms_on_masks(masks: List[np.ndarray], iou_thresh: float = 0.7) -> List[np.ndarray]:
    """간단한 마스크 NMS: 큰 마스크부터, 많이 겹치면 버림"""
    areas = [m.sum() for m in masks]
    order = np.argsort(areas)[::-1]
    kept = []
    for idx in order:
        m = masks[idx]
        if all(iou(m, km) < iou_thresh for km in kept):
            kept.append(m)
    return kept


# =========================
# SAM: 자동 마스크 생성
# =========================
@dataclass
class AutoMask:
    mask: np.ndarray       # HxW bool
    score: float
    area: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h


def load_sam(checkpoint: str, model_type: str):
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to("cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")
    return SamAutomaticMaskGenerator(sam, **MASK_GEN_KW)


def generate_auto_masks(bgr: np.ndarray, mag: SamAutomaticMaskGenerator) -> List[AutoMask]:
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    raw = mag.generate(img_rgb)  # list of dicts
    H, W = bgr.shape[:2]
    img_area = H * W

    masks = []
    for d in raw:
        m = d["segmentation"].astype(bool)
        area = int(m.sum())
        # 너무 작거나 너무 큰 마스크는 제외 (0.2% ~ 15% 사이)
        if area < img_area * 0.002 or area > img_area * 0.15:
            continue
        x, y, w, h = d["bbox"]
        masks.append(AutoMask(mask=m, score=float(d.get("predicted_iou", 0.0)), area=area, bbox=(x, y, w, h)))

    # NMS로 겹침 정리
    masks_sorted = sorted(masks, key=lambda z: z.area, reverse=True)
    masks_only = [m.mask for m in masks_sorted]
    kept_masks = nms_on_masks(masks_only, iou_thresh=0.6)

    # kept 순서대로 AutoMask 매칭
    kept = []
    used = set()
    for km in kept_masks:
        for am in masks_sorted:
            if id(am) in used:
                continue
            if am.mask.shape == km.shape and (am.mask ^ km).sum() == 0:
                kept.append(am)
                used.add(id(am))
                break

    return kept


# =========================
# 편집(차이 만들기)
# =========================
def inpaint_region(bgr: np.ndarray, mask: np.ndarray, radius: int = 5) -> np.ndarray:
    """해당 영역 지우기(인페인트)"""
    mask_u8 = (mask.astype(np.uint8) * 255)
    # 약간 확장해서 이음새 제거
    kernel = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(mask_u8, kernel, iterations=1)
    result = cv2.inpaint(bgr, dil, radius, cv2.INPAINT_TELEA)
    return result

def hsv_shift_region(bgr: np.ndarray, mask: np.ndarray, dh: int = 10, ds: int = 15, dv: int = 0) -> np.ndarray:
    """색상/채도/명도 살짝 변경"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = mask.astype(bool)
    h, s, v = cv2.split(hsv)
    h[m] = (h[m].astype(int) + dh) % 180
    s[m] = np.clip(s[m].astype(int) + ds, 0, 255).astype(np.uint8)
    v[m] = np.clip(v[m].astype(int) + dv, 0, 255).astype(np.uint8)
    hsv2 = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

def mirror_region(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """마스크 ROI만 좌우 반전"""
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    roi = bgr[y:y+h, x:x+w].copy()
    roi_mask = mask[y:y+h, x:x+w]
    flipped = cv2.flip(roi, 1)
    out = bgr.copy()
    out[y:y+h, x:x+w][roi_mask] = flipped[roi_mask]
    return out

def translate_region(bgr: np.ndarray, mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """
    객체를 조금 옮기기: (1) 원래 자리는 인페인트, (2) 잘라낸 객체를 새 위치에 붙이기.
    """
    H, W = bgr.shape[:2]
    # (1) 원래 자리 인페인트
    bg = inpaint_region(bgr, mask, radius=5)

    # 붙일 객체 잘라내기
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return bgr
    x0, x1 = xs.min(), xs.max()+1
    y0, y1 = ys.min(), ys.max()+1
    cut = bgr[y0:y1, x0:x1].copy()
    cut_mask = mask[y0:y1, x0:x1].astype(np.uint8)

    # 이동 좌표
    tx = np.clip(x0 + dx, 0, max(0, W - (x1 - x0)))
    ty = np.clip(y0 + dy, 0, max(0, H - (y1 - y0)))

    out = bg.copy()
    region = out[ty:ty+(y1-y0), tx:tx+(x1-x0)]
    # 가장자리 부자연스러움 줄이기 위해 약간 blur된 알파 사용
    alpha = cv2.GaussianBlur(cut_mask.astype(np.float32), (5,5), 0)
    alpha = (alpha / (alpha.max() + 1e-6))[:, :, None]

    region = region.astype(np.float32)
    cut_f = cut.astype(np.float32)
    comp = cut_f * alpha + region * (1 - alpha)
    out[ty:ty+(y1-y0), tx:tx+(x1-x0)] = comp.astype(np.uint8)
    return out

def draw_answer_boxes(bgr: np.ndarray, diff_masks: List[np.ndarray]) -> np.ndarray:
    """정답 표시용: 변경 영역들에 사각형"""
    out = bgr.copy()
    for m in diff_masks:
        x, y, w, h = cv2.boundingRect(m.astype(np.uint8))
        cv2.rectangle(out, (x,y), (x+w, y+h), (0,0,255), 3)
    return out

def union_masks(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    return np.logical_or(m1, m2)

# 편집 타입
EDIT_FUNCS = ("erase", "hsv", "mirror", "move")


# =========================
# LLM 플래너(선택)
# =========================
def plan_with_llm(mask_summaries: List[Dict], n_changes: int, seed: int = 0) -> List[Dict]:
    """
    OpenAI API로 '어떤 마스크에 어떤 편집을 할지' 계획을 받는다.
    mask_summaries: [{"idx": 3, "area_pct": 2.1, "pos":"top-left"}, ...]
    return: [{"idx": 3, "edit":"hsv"}, ...]
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # API키 없으면 랜덤 계획
        random.seed(seed)
        choices = random.sample(mask_summaries, k=min(n_changes, len(mask_summaries)))
        plan = []
        for c in choices:
            plan.append({"idx": c["idx"], "edit": random.choice(list(EDIT_FUNCS))})
        return plan

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        sys = "You plan subtle edits for a spot-the-difference puzzle. Prefer small, tricky changes."
        user = f"Pick {n_changes} indices and an edit from {EDIT_FUNCS}. Masks: {mask_summaries}"
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0.2
        )
        text = resp.choices[0].message.content
        # 아주 단순 파서: {"idx":int,"edit":"hsv"} 형태만 추출 시도
        import json, re
        js = None
        try:
            js = json.loads(text)
        except:
            # 코드블록 등에서 JSON을 추출
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                js = json.loads(m.group(0))
        if isinstance(js, list):
            # 유효성 보정
            plan = []
            for it in js[:n_changes]:
                if isinstance(it, dict) and "idx" in it and "edit" in it and it["edit"] in EDIT_FUNCS:
                    plan.append({"idx": int(it["idx"]), "edit": it["edit"]})
            if plan:
                return plan
    except Exception:
        pass

    # 실패시 랜덤
    random.seed(seed)
    choices = random.sample(mask_summaries, k=min(n_changes, len(mask_summaries)))
    return [{"idx": c["idx"], "edit": random.choice(list(EDIT_FUNCS))} for c in choices]


# =========================
# 메인 파이프라인
# =========================
def make_puzzle(
    bgr: np.ndarray,
    n_changes: int = 5,
    difficulty: str = "normal",
    seed: int = 0,
    use_llm: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    return: (puzzle_bgr, answer_bgr, diff_masks)
    """
    random.seed(seed)
    np.random.seed(seed)

    mag = load_sam(SAM_CHECKPOINT, SAM_MODEL_TYPE)
    auto_masks = generate_auto_masks(bgr, mag)

    if len(auto_masks) == 0:
        raise RuntimeError("적절한 객체 마스크를 찾지 못했습니다. (사진을 바꾸거나 파라미터를 조정하세요)")

    # 난이도별 편집 폭
    if difficulty == "easy":
        dh_range = (-6, 6); ds_range = (-8, 8); move_px = 6
    elif difficulty == "hard":
        dh_range = (-18, 18); ds_range = (-25, 25); move_px = 18
    else:  # normal
        dh_range = (-12, 12); ds_range = (-16, 16); move_px = 12

    # 요약(LLM용)
    H, W = bgr.shape[:2]
    msumm = []
    for i, am in enumerate(auto_masks):
        x, y, w, h = am.bbox
        cx = (x + x + w) / 2 / W
        cy = (y + y + h) / 2 / H
        pos = []
        pos.append("top" if cy < 0.33 else "bottom" if cy > 0.66 else "middle")
        pos.append("left" if cx < 0.33 else "right" if cx > 0.66 else "center")
        msumm.append({"idx": i, "area_pct": round(100.0 * am.area / (H*W), 2), "pos": "-".join(pos)})

    plan = plan_with_llm(msumm, n_changes, seed) if use_llm else None
    chosen = []
    if plan:
        # 계획된 인덱스만
        valid_idx = {p["idx"] for p in plan if 0 <= p["idx"] < len(auto_masks)}
        chosen = [auto_masks[i] for i in valid_idx]
    else:
        # 랜덤 선택
        k = min(n_changes, len(auto_masks))
        chosen = random.sample(auto_masks, k=k)

    out = bgr.copy()
    diff_masks: List[np.ndarray] = []

    for am in chosen:
        m = am.mask.copy()
        edit = None
        if plan:
            # 해당 마스크의 편집 찾기
            found = [p for p in plan if p["idx"] == auto_masks.index(am)]
            if found:
                edit = found[0]["edit"]

        if edit is None:
            edit = random.choice(list(EDIT_FUNCS))

        # 편집 적용
        if edit == "erase":
            new_img = inpaint_region(out, m, radius=5)
            changed = m.copy()

        elif edit == "hsv":
            dh = random.randint(*dh_range)
            ds = random.randint(*ds_range)
            dv = random.randint(-8, 8)
            new_img = hsv_shift_region(out, m, dh=dh, ds=ds, dv=dv)
            changed = m.copy()

        elif edit == "mirror":
            new_img = mirror_region(out, m)
            changed = m.copy()

        elif edit == "move":
            dx = random.randint(-move_px, move_px)
            dy = random.randint(-move_px, move_px)
            new_img = translate_region(out, m, dx=dx, dy=dy)
            # 이동은 원래 위치 + 새 위치 모두 차이
            moved_mask = np.zeros_like(m, dtype=bool)
            # 새 위치 대략 bbox 이동으로 근사한 솔루션 마스크 (충분히 정답표시용)
            x, y, w, h = cv2.boundingRect(m.astype(np.uint8))
            tx = np.clip(x + dx, 0, out.shape[1]-w)
            ty = np.clip(y + dy, 0, out.shape[0]-h)
            moved_mask[ty:ty+h, tx:tx+w] = True
            changed = union_masks(m, moved_mask)

        else:
            new_img = out
            changed = m.copy()

        out = new_img
        diff_masks.append(changed)

    answer = draw_answer_boxes(out, diff_masks)
    return out, answer, diff_masks


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="틀린그림 찾기 자동 생성기", layout="wide")
st.title("🧩 틀린그림 찾기 자동 생성기 (SAM 기반)")

with st.sidebar:
    st.header("옵션")
    n_changes = st.slider("변경 개수", 1, 10, 5)
    difficulty = st.selectbox("난이도", ["easy", "normal", "hard"], index=1)
    seed = st.number_input("랜덤 시드", value=0, step=1)
    use_llm = st.checkbox("LLM으로 편집 계획 세우기 (선택)", value=False)
    st.markdown("---")
    st.write("**SAM Checkpoint**")
    st.code(SAM_CHECKPOINT)
    st.caption("경로가 올바른지 확인하세요.")

uploaded = st.file_uploader("이미지 업로드 (jpg/png)", type=["jpg", "jpeg", "png"])

col1, col2, col3 = st.columns(3)

if uploaded:
    pil = Image.open(uploaded)
    bgr = pil_to_bgr(pil)
    col1.subheader("원본")
    col1.image(pil, use_column_width=True)

    try:
        with st.spinner("객체 인식 및 교묘한 변경 생성 중..."):
            puzzle_bgr, answer_bgr, diff_masks = make_puzzle(
                bgr, n_changes=n_changes, difficulty=difficulty, seed=seed, use_llm=use_llm
            )

        col2.subheader("문제(달라진 그림)")
        col2.image(bgr_to_pil(puzzle_bgr), use_column_width=True)

        col3.subheader("정답(변경 영역 표시)")
        col3.image(bgr_to_pil(answer_bgr), use_column_width=True)

        # 다운로드
        buf_p = io.BytesIO()
        bgr_to_pil(puzzle_bgr).save(buf_p, format="PNG")
        st.download_button("문제 이미지 다운로드", data=buf_p.getvalue(), file_name="puzzle.png", mime="image/png")

        buf_a = io.BytesIO()
        bgr_to_pil(answer_bgr).save(buf_a, format="PNG")
        st.download_button("정답 이미지 다운로드", data=buf_a.getvalue(), file_name="answer.png", mime="image/png")

    except Exception as e:
        st.error(f"생성 실패: {e}")
else:
    st.info("좌측에서 옵션을 조정하고, 위에 이미지를 업로드하세요.")
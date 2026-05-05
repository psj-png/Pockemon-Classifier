"""
app.py — PokéInsight Streamlit Demo

탭 구성:
  1. 도감 스캔  — 포켓몬 카드 UI + 타입 뱃지 예측 결과
  2. GradCAM   — 원본 | 히트맵 나란히 비교
  3. 실험 랩   — 4개 실험 성능 대시보드 + 학습 곡선
"""
from __future__ import annotations

import base64
import json
import os
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image

from src.config import CKPT_DIR, EXPERIMENTS, RESULT_DIR
from src.dataset import get_eval_transform
from src.gradcam import GradCAM, get_target_layer, overlay_heatmap
from src.model import build_model
from src.tta import predict_single_standard, predict_single_tta

# ──────────────────────────────────────────────────────────────
# 포켓몬 타입 매핑 & 색상 시스템
# ──────────────────────────────────────────────────────────────
POKEMON_TYPES: dict[str, str] = {
    "Abra":"psychic","Aerodactyl":"rock","Alakazam":"psychic",
    "Arcanine":"fire","Articuno":"ice","Beedrill":"bug",
    "Bellsprout":"grass","Blastoise":"water","Bulbasaur":"grass",
    "Butterfree":"bug","Caterpie":"bug","Chansey":"normal",
    "Charizard":"fire","Charmander":"fire","Charmeleon":"fire",
    "Clefable":"normal","Clefairy":"normal","Cloyster":"water",
    "Cubone":"ground","Dewgong":"water","Diglett":"ground",
    "Ditto":"normal","Dodrio":"normal","Doduo":"normal",
    "Dragonair":"dragon","Dragonite":"dragon","Dratini":"dragon",
    "Drowzee":"psychic","Dugtrio":"ground","Eevee":"normal",
    "Ekans":"poison","Electabuzz":"electric","Electrode":"electric",
    "Exeggcute":"grass","Exeggutor":"grass","Farfetchd":"normal",
    "Fearow":"normal","Flareon":"fire","Gastly":"ghost",
    "Gengar":"ghost","Geodude":"rock","Gloom":"grass",
    "Golbat":"poison","Goldeen":"water","Golduck":"water",
    "Golem":"rock","Graveler":"rock","Grimer":"poison",
    "Growlithe":"fire","Gyarados":"water","Haunter":"ghost",
    "Hitmonchan":"fighting","Hitmonlee":"fighting","Horsea":"water",
    "Hypno":"psychic","Ivysaur":"grass","Jigglypuff":"normal",
    "Jolteon":"electric","Jynx":"psychic","Kabuto":"rock",
    "Kabutops":"rock","Kadabra":"psychic","Kakuna":"bug",
    "Kangaskhan":"normal","Kingler":"water","Koffing":"poison",
    "Krabby":"water","Lapras":"water","Lickitung":"normal",
    "Machamp":"fighting","Machoke":"fighting","Machop":"fighting",
    "Magikarp":"water","Magmar":"fire","Magnemite":"electric",
    "Magneton":"electric","Mankey":"fighting","Marowak":"ground",
    "Meowth":"normal","Metapod":"bug","Mew":"psychic",
    "Mewtwo":"psychic","Moltres":"fire","MrMime":"psychic",
    "Muk":"poison","Nidoking":"poison","Nidoqueen":"poison",
    "NidoranF":"poison","NidoranM":"poison","Nidorina":"poison",
    "Nidorino":"poison","Ninetales":"fire","Oddish":"grass",
    "Omanyte":"rock","Omastar":"rock","Onix":"rock",
    "Paras":"bug","Parasect":"bug","Persian":"normal",
    "Pidgeot":"normal","Pidgeotto":"normal","Pidgey":"normal",
    "Pikachu":"electric","Pinsir":"bug","Poliwag":"water",
    "Poliwhirl":"water","Poliwrath":"water","Ponyta":"fire",
    "Porygon":"normal","Primeape":"fighting","Psyduck":"water",
    "Raichu":"electric","Rapidash":"fire","Raticate":"normal",
    "Rattata":"normal","Rhydon":"ground","Rhyhorn":"ground",
    "Sandshrew":"ground","Sandslash":"ground","Scyther":"bug",
    "Seadra":"water","Seaking":"water","Seel":"water",
    "Shellder":"water","Slowbro":"water","Slowpoke":"water",
    "Snorlax":"normal","Spearow":"normal","Squirtle":"water",
    "Starmie":"water","Staryu":"water","Tangela":"grass",
    "Tauros":"normal","Tentacool":"water","Tentacruel":"water",
    "Vaporeon":"water","Venomoth":"bug","Venonat":"bug",
    "Venusaur":"grass","Victreebel":"grass","Vileplume":"grass",
    "Voltorb":"electric","Vulpix":"fire","Wartortle":"water",
    "Weedle":"bug","Weepinbell":"grass","Weezing":"poison",
    "Wigglytuff":"normal","Zapdos":"electric","Zubat":"poison",
}

TYPE_COLORS: dict[str, dict] = {
    "fire"    : {"bg":"#FF6B35","fg":"#fff","glow":"rgba(255,107,53,0.45)"},
    "water"   : {"bg":"#4DA6FF","fg":"#fff","glow":"rgba(77,166,255,0.45)"},
    "grass"   : {"bg":"#5DBE6E","fg":"#fff","glow":"rgba(93,190,110,0.45)"},
    "electric": {"bg":"#F7D51D","fg":"#1a1a1a","glow":"rgba(247,213,29,0.45)"},
    "psychic" : {"bg":"#FF5B9A","fg":"#fff","glow":"rgba(255,91,154,0.45)"},
    "ice"     : {"bg":"#74CEC0","fg":"#fff","glow":"rgba(116,206,192,0.45)"},
    "dragon"  : {"bg":"#6A4AFF","fg":"#fff","glow":"rgba(106,74,255,0.45)"},
    "ghost"   : {"bg":"#7B62A3","fg":"#fff","glow":"rgba(123,98,163,0.45)"},
    "rock"    : {"bg":"#C9BB8A","fg":"#1a1a1a","glow":"rgba(201,187,138,0.45)"},
    "ground"  : {"bg":"#D97845","fg":"#fff","glow":"rgba(217,120,69,0.45)"},
    "bug"     : {"bg":"#9DC130","fg":"#fff","glow":"rgba(157,193,48,0.45)"},
    "poison"  : {"bg":"#A95FA0","fg":"#fff","glow":"rgba(169,95,160,0.45)"},
    "fighting": {"bg":"#D94020","fg":"#fff","glow":"rgba(217,64,32,0.45)"},
    "normal"  : {"bg":"#A0A29F","fg":"#fff","glow":"rgba(160,162,159,0.45)"},
}


def _type_color(name: str) -> dict:
    t = POKEMON_TYPES.get(name, "normal")
    return TYPE_COLORS.get(t, TYPE_COLORS["normal"])


def _img_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _np_b64(arr: np.ndarray) -> str:
    return _img_b64(Image.fromarray(arr))


# ──────────────────────────────────────────────────────────────
# 페이지 설정 & 전역 스타일
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PokéInsight",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Outfit:wght@300;400;600;700&display=swap');
.stApp { background:#0d0d12; color:#e0e0ea; }
section[data-testid="stSidebar"] { display:none; }
body, p, div { font-family:'Outfit', sans-serif; }
[data-testid="stFileUploader"] {
    background:#15151e !important;
    border:2px dashed #2a2a3c !important;
    border-radius:14px !important;
}
.stTabs [role="tab"] {
    font-family:'Outfit',sans-serif; font-weight:600;
    color:#555; background:transparent; border:none;
    padding:10px 22px; font-size:14px;
}
.stTabs [role="tab"][aria-selected="true"] {
    color:#e0e0ea; border-bottom:2px solid #FF6B35;
}
.stTabs [data-baseweb="tab-list"] {
    background:transparent; border-bottom:1px solid #1e1e2c;
}
.stSelectbox > div > div {
    background:#15151e !important;
    border-color:#2a2a3c !important;
    color:#e0e0ea !important;
    border-radius:10px !important;
}
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-thumb { background:#2a2a3c; border-radius:3px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# 헤더
# ──────────────────────────────────────────────────────────────
components.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Outfit:wght@700&display=swap');
.hdr { display:flex; align-items:center; gap:18px; padding:28px 0 20px; }
.pokeball {
    width:48px; height:48px; border-radius:50%;
    background:linear-gradient(180deg,#FF6B35 50%,#fff 50%);
    border:3px solid #fff; position:relative; flex-shrink:0;
    box-shadow:0 0 22px rgba(255,107,53,.55);
}
.pokeball::after {
    content:''; position:absolute; left:50%; top:50%;
    transform:translate(-50%,-50%);
    width:13px; height:13px; border-radius:50%;
    background:#fff; border:3px solid #222;
}
.divline {
    position:absolute; left:0; top:calc(50% - 1px);
    width:100%; height:2px; background:#222;
}
h1.t { font-family:'Press Start 2P',monospace; font-size:20px; color:#fff; margin:0; }
h1.t span { color:#FF6B35; }
p.s { font-family:'Outfit',sans-serif; font-size:13px; color:#555; margin:4px 0 0; }
</style>
<div class="hdr">
  <div class="pokeball"><div class="divline"></div></div>
  <div>
    <h1 class="t">Poké<span>Insight</span></h1>
    <p class="s">Transfer Learning · TTA Experiment Analysis · GradCAM Visualization</p>
  </div>
</div>
""", height=100)


# ──────────────────────────────────────────────────────────────
# 모델 로딩 (캐시)
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_ckpt(exp_name: str):
    path = os.path.join(CKPT_DIR, f"{exp_name}.pth")
    if not os.path.exists(path):
        return None, None, None, None
    ckpt  = torch.load(path, map_location="cpu")
    model = build_model(
        ckpt["backbone"], pretrained=False,
        num_classes=len(ckpt["class_names"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["class_names"], ckpt["backbone"], ckpt.get("use_tta", False)


_eval_tfm = get_eval_transform()


def _infer(model, class_names: list[str], img: Image.Image,
           use_tta: bool) -> np.ndarray:
    device = next(model.parameters()).device
    if use_tta:
        return predict_single_tta(model, img, device)
    return predict_single_standard(model, img, device, _eval_tfm)


# ──────────────────────────────────────────────────────────────
# 탭 구성
# ──────────────────────────────────────────────────────────────
tab_scan, tab_cam, tab_lab = st.tabs(
    ["📋  도감 스캔", "🔥  GradCAM", "📊  실험 랩"]
)


# ══════════════════════════════════════════════════════════════
# 탭 1 — 도감 스캔
# ══════════════════════════════════════════════════════════════
with tab_scan:
    col_up, col_sel = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader(
            "포켓몬 이미지 업로드",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
    with col_sel:
        sel_scan = st.selectbox(
            "모델",
            list(EXPERIMENTS.keys()),
            format_func=lambda x: EXPERIMENTS[x]["description"].split("|")[0].strip(),
            key="sel_scan",
            label_visibility="collapsed",
        )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        model, class_names, backbone, use_tta = load_ckpt(sel_scan)

        if model is None:
            st.warning("⚠️ 체크포인트 없음 — `python run_experiments.py` 를 먼저 실행하세요.")
        else:
            probs      = _infer(model, class_names, img, use_tta)
            top_idx    = np.argsort(probs)[::-1][:5]
            top1_name  = class_names[top_idx[0]]
            top1_prob  = float(probs[top_idx[0]])
            c          = _type_color(top1_name)
            img_b64    = _img_b64(img.resize((300, 300)))

            dex_no = (list(POKEMON_TYPES.keys()).index(top1_name) + 1
                      if top1_name in POKEMON_TYPES else 0)

            # 후보 카드 HTML
            cands_html = ""
            for rank, i in enumerate(top_idx[1:], 2):
                nm = class_names[i]
                pb = float(probs[i])
                cc = _type_color(nm)
                cands_html += f"""
                <div class="cand" style="animation-delay:{(rank-1)*0.06}s">
                  <span class="cbadge"
                    style="background:{cc['bg']};color:{cc['fg']}">
                    {POKEMON_TYPES.get(nm,'normal').upper()}
                  </span>
                  <span class="cname">{nm}</span>
                  <span class="cprob">{pb*100:.1f}%</span>
                </div>"""

            tta_badge = (
                '<span class="tta-on">TTA ON</span>' if use_tta
                else '<span class="tta-off">TTA OFF</span>'
            )

            components.html(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Outfit:wght@300;400;600;700&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
@keyframes cardIn{{
  from{{opacity:0;transform:translateY(20px) scale(.97)}}
  to{{opacity:1;transform:none}}
}}
@keyframes barGrow{{
  from{{transform:scaleX(0)}}
  to{{transform:scaleX(1)}}
}}

.wrap{{display:flex;gap:24px;padding:8px 0 20px;align-items:flex-start}}

/* ── 도감 카드 */
.card{{
  width:290px;flex-shrink:0;background:#15151e;
  border-radius:18px;border:1px solid #22223a;overflow:hidden;
  box-shadow:0 0 36px {c['glow']};
  animation:cardIn .5s cubic-bezier(.22,.68,0,1.2) both;
}}
.card-head{{
  background:linear-gradient(135deg,{c['bg']}22,{c['bg']}40);
  border-bottom:1px solid {c['bg']}44;
  padding:13px 16px 9px;
  display:flex;justify-content:space-between;align-items:center;
}}
.dex-no{{
  font-family:'Press Start 2P',monospace;
  font-size:8px;color:{c['bg']};opacity:.8
}}
.conf-chip{{
  font-family:'Outfit',sans-serif;font-size:11px;font-weight:700;
  background:{c['bg']};color:{c['fg']};
  padding:3px 9px;border-radius:99px;
}}
.card-img{{
  width:100%;height:210px;object-fit:contain;
  background:#0d0d14;display:block;padding:14px;
}}
.card-body{{padding:14px 16px 18px}}
.pname{{
  font-family:'Press Start 2P',monospace;font-size:12px;
  color:#fff;margin-bottom:9px;line-height:1.5;
}}
.tbadge{{
  display:inline-block;font-family:'Outfit',sans-serif;
  font-size:10px;font-weight:700;letter-spacing:1.4px;
  text-transform:uppercase;padding:3px 11px;border-radius:99px;
  background:{c['bg']};color:{c['fg']};
  box-shadow:0 2px 10px {c['glow']};margin-bottom:12px;
}}
.bar-label{{
  font-family:'Outfit',sans-serif;font-size:10px;color:#444;
  text-transform:uppercase;letter-spacing:.8px;margin-bottom:5px;
}}
.bar-track{{height:5px;background:#1e1e2c;border-radius:99px;overflow:hidden}}
.bar-fill{{
  height:100%;width:{top1_prob*100:.1f}%;
  background:linear-gradient(90deg,{c['bg']}99,{c['bg']});
  border-radius:99px;transform-origin:left;
  animation:barGrow .8s cubic-bezier(.22,.68,0,1.2) .3s both;
}}
.conf-num{{
  font-family:'Outfit',sans-serif;font-size:26px;font-weight:700;
  color:{c['bg']};margin-top:5px;line-height:1;
}}
.conf-num small{{font-size:13px;color:#444;font-weight:400}}
.tta-on{{
  display:inline-block;font-family:'Outfit',sans-serif;
  font-size:9px;font-weight:700;letter-spacing:1px;
  padding:2px 7px;border-radius:99px;
  background:#6A4AFF22;color:#6A4AFF;border:1px solid #6A4AFF44;
  margin-top:8px;
}}
.tta-off{{
  display:inline-block;font-family:'Outfit',sans-serif;
  font-size:9px;font-weight:700;letter-spacing:1px;
  padding:2px 7px;border-radius:99px;
  background:#33333344;color:#666;border:1px solid #33333366;
  margin-top:8px;
}}

/* ── 우측 후보 패널 */
.right{{flex:1;display:flex;flex-direction:column;gap:10px}}
.panel-lbl{{
  font-family:'Outfit',sans-serif;font-size:10px;
  text-transform:uppercase;letter-spacing:1.4px;color:#3a3a4c;margin-bottom:2px;
}}
.cand{{
  background:#15151e;border:1px solid #1e1e2c;border-radius:12px;
  padding:11px 14px;display:flex;align-items:center;gap:11px;
  animation:cardIn .4s cubic-bezier(.22,.68,0,1.2) both;
}}
.cand:hover{{background:#1c1c28}}
.cbadge{{
  font-family:'Outfit',sans-serif;font-size:8px;font-weight:700;
  letter-spacing:1px;padding:3px 8px;border-radius:99px;
  flex-shrink:0;min-width:54px;text-align:center;
}}
.cname{{font-family:'Outfit',sans-serif;font-size:14px;font-weight:600;color:#bbb;flex:1}}
.cprob{{font-family:'Outfit',sans-serif;font-size:15px;font-weight:700;color:#444}}
</style>

<div class="wrap">
  <div class="card">
    <div class="card-head">
      <span class="dex-no">No.{str(dex_no).zfill(3)}</span>
      <span class="conf-chip">{top1_prob*100:.1f}%</span>
    </div>
    <img class="card-img" src="data:image/png;base64,{img_b64}"/>
    <div class="card-body">
      <div class="pname">{top1_name}</div>
      <div class="tbadge">{POKEMON_TYPES.get(top1_name,'NORMAL').upper()}</div>
      <div class="bar-label">Confidence</div>
      <div class="bar-track"><div class="bar-fill"></div></div>
      <div class="conf-num">{top1_prob*100:.1f}<small>%</small></div>
      {tta_badge}
    </div>
  </div>
  <div class="right">
    <div class="panel-lbl">다른 후보</div>
    {cands_html}
  </div>
</div>
""", height=510)


# ══════════════════════════════════════════════════════════════
# 탭 2 — GradCAM
# ══════════════════════════════════════════════════════════════
with tab_cam:
    sel_cam = st.selectbox(
        "GradCAM 모델",
        list(EXPERIMENTS.keys()),
        format_func=lambda x: EXPERIMENTS[x]["description"],
        key="sel_cam",
        label_visibility="collapsed",
    )
    gc_up = st.file_uploader(
        "GradCAM 분석 이미지",
        type=["jpg", "jpeg", "png"],
        key="gc",
        label_visibility="collapsed",
    )

    if gc_up:
        img_gc = Image.open(gc_up).convert("RGB")
        model, class_names, backbone, _ = load_ckpt(sel_cam)

        if model is None:
            st.warning("⚠️ 체크포인트 없음")
        else:
            tensor     = _eval_tfm(img_gc).unsqueeze(0)
            cam_obj    = GradCAM(model, get_target_layer(model, backbone))
            cam, pred_idx = cam_obj(tensor)
            overlay    = overlay_heatmap(img_gc, cam)

            pred_name  = class_names[pred_idx]
            c          = _type_color(pred_name)
            orig_b64   = _img_b64(img_gc.resize((280, 280)))
            ovl_b64    = _np_b64(cv2.resize(overlay, (280, 280)))

            components.html(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Outfit:wght@400;600;700&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
@keyframes fu{{from{{opacity:0;transform:translateY(14px)}}to{{opacity:1;transform:none}}}}
.row{{display:flex;gap:20px;padding:8px 0 20px;align-items:flex-start}}
.panel{{
  background:#15151e;border:1px solid #1e1e2c;
  border-radius:14px;overflow:hidden;flex:1;
  animation:fu .4s ease both;
}}
.panel:nth-child(2){{animation-delay:.1s}}
.plbl{{
  font-family:'Outfit',sans-serif;font-size:10px;
  text-transform:uppercase;letter-spacing:1.4px;color:#444;
  padding:13px 16px 9px;border-bottom:1px solid #1a1a28;
}}
.panel img{{width:100%;display:block;background:#0d0d14}}
.result{{
  background:#15151e;border:1px solid {c['bg']}33;
  border-radius:14px;padding:20px;width:210px;flex-shrink:0;
  box-shadow:0 0 26px {c['glow']};
  animation:fu .4s ease .2s both;
}}
.rlbl{{
  font-family:'Outfit',sans-serif;font-size:10px;
  text-transform:uppercase;letter-spacing:1.4px;
  color:#333;margin-bottom:10px;
}}
.rname{{
  font-family:'Press Start 2P',monospace;font-size:10px;
  color:#fff;line-height:1.6;margin-bottom:10px;
}}
.rtbadge{{
  display:inline-block;font-family:'Outfit',sans-serif;
  font-size:9px;font-weight:700;letter-spacing:1.4px;
  text-transform:uppercase;padding:3px 10px;border-radius:99px;
  background:{c['bg']};color:{c['fg']};
  box-shadow:0 2px 10px {c['glow']};margin-bottom:16px;
}}
.rdesc{{
  font-family:'Outfit',sans-serif;font-size:11px;color:#444;
  line-height:1.7;border-top:1px solid #1e1e2c;padding-top:14px;
}}
.rdesc b{{color:#FF6B35}}
</style>
<div class="row">
  <div class="panel">
    <div class="plbl">원본 이미지</div>
    <img src="data:image/png;base64,{orig_b64}"/>
  </div>
  <div class="panel">
    <div class="plbl">GradCAM 히트맵</div>
    <img src="data:image/png;base64,{ovl_b64}"/>
  </div>
  <div class="result">
    <div class="rlbl">모델 판단</div>
    <div class="rname">{pred_name}</div>
    <div class="rtbadge">{POKEMON_TYPES.get(pred_name,'NORMAL').upper()}</div>
    <div class="rdesc">
      <b>붉은 영역</b>이 모델이
      <b>{pred_name}</b> 으로 판단하는 데
      가장 크게 기여한 부분입니다.<br><br>
      사용 모델:<br>
      {EXPERIMENTS[sel_cam]['description']}
    </div>
  </div>
</div>
""", height=390)


# ══════════════════════════════════════════════════════════════
# 탭 3 — 실험 랩 대시보드
# ══════════════════════════════════════════════════════════════
with tab_lab:
    spath = os.path.join(RESULT_DIR, "summary.json")

    if not os.path.exists(spath):
        st.info(
            "`python run_experiments.py` 를 먼저 실행하면 "
            "결과가 여기에 표시됩니다."
        )
    else:
        with open(spath) as f:
            summary = json.load(f)

        best = max(summary, key=lambda x: summary[x]["accuracy"])

        # ── 요약 카드 4개
        avgs = {
            mk: float(np.mean([v[mk] for v in summary.values()]))
            for mk in ["accuracy", "precision", "recall", "f1"]
        }
        items = [
            ("Best Acc",   f"{summary[best]['accuracy']*100:.1f}%",  "#FF6B35"),
            ("Best F1",    f"{summary[best]['f1']*100:.1f}%",        "#5DBE6E"),
            ("Avg Recall", f"{avgs['recall']*100:.1f}%",             "#4DA6FF"),
            ("실험 수",    str(len(summary)),                         "#F7D51D"),
        ]
        cards_html = "".join(
            f"<div class='mc' style='animation-delay:{i*.05}s'>"
            f"<div class='mv' style='color:{color}'>{val}</div>"
            f"<div class='ml'>{lbl}</div></div>"
            for i, (lbl, val, color) in enumerate(items)
        )

        # ── 테이블 행
        rows_html = ""
        for name, m in summary.items():
            is_best = name == best
            cfg     = EXPERIMENTS.get(name, {})
            tta_tag = "✓ TTA" if cfg.get("use_tta") else "—"
            desc    = cfg.get("description", name)
            star    = "🥇 " if is_best else ""
            rows_html += (
                f"<tr class=\"{'br' if is_best else ''}\">"
                f"<td class='et'>{star}{desc}</td>"
                f"<td class='tta'>{tta_tag}</td>"
                f"<td class='nt' style='color:#FF6B35'>{m['accuracy']*100:.2f}%</td>"
                f"<td class='nt' style='color:#4DA6FF'>{m['precision']*100:.2f}%</td>"
                f"<td class='nt' style='color:#5DBE6E'>{m['recall']*100:.2f}%</td>"
                f"<td class='nt' style='color:#F7D51D'>{m['f1']*100:.2f}%</td>"
                f"</tr>"
            )

        components.html(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
@keyframes fu{{from{{opacity:0;transform:translateY(10px)}}to{{opacity:1;transform:none}}}}
.wrap{{padding:12px 0 24px}}
.mgrid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}}
.mc{{
  background:#15151e;border:1px solid #1e1e2c;border-radius:12px;
  padding:16px 18px;animation:fu .4s ease both;
}}
.mv{{font-family:'Outfit',sans-serif;font-size:26px;font-weight:700;line-height:1;margin-bottom:5px}}
.ml{{font-family:'Outfit',sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:1.2px;color:#3a3a4c}}
.slbl{{font-family:'Outfit',sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:1.4px;color:#333;margin-bottom:10px}}
table{{width:100%;border-collapse:collapse;font-family:'Outfit',sans-serif}}
thead tr{{border-bottom:1px solid #1e1e2c}}
th{{font-size:9px;text-transform:uppercase;letter-spacing:1.2px;color:#333;padding:7px 10px;text-align:left;font-weight:600}}
th.nth{{text-align:right}}
td{{padding:11px 10px;font-size:13px}}
.et{{color:#999;font-size:12px}}
.tta{{font-size:11px;color:#6A4AFF;font-weight:600;text-align:center}}
.nt{{text-align:right;font-weight:700;font-size:14px}}
tbody tr{{border-bottom:1px solid #16161e;transition:background .15s}}
tbody tr:hover{{background:#18182a}}
.br{{background:#1a1826 !important}}
.br .et{{color:#e0e0ea;font-weight:600}}
</style>
<div class="wrap">
  <div class="mgrid">{cards_html}</div>
  <div class="slbl">실험별 성능 비교</div>
  <table>
    <thead>
      <tr>
        <th>실험 구성</th>
        <th style="text-align:center">TTA</th>
        <th class="nth" style="color:#FF6B35">Accuracy</th>
        <th class="nth" style="color:#4DA6FF">Precision</th>
        <th class="nth" style="color:#5DBE6E">Recall</th>
        <th class="nth" style="color:#F7D51D">F1</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
""", height=360)

        # ── 학습 곡선
        st.markdown(
            '<p style="font-family:Outfit,sans-serif;font-size:10px;'
            'text-transform:uppercase;letter-spacing:1.4px;color:#333;'
            'margin:20px 0 10px">학습 곡선</p>',
            unsafe_allow_html=True,
        )
        curve_cols = st.columns(2)
        for i, name in enumerate(EXPERIMENTS):
            cp = os.path.join(RESULT_DIR, "curves", f"{name}.png")
            if os.path.exists(cp):
                curve_cols[i % 2].image(
                    cp,
                    caption=EXPERIMENTS[name]["description"],
                    use_column_width=True,
                )

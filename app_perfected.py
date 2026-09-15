"""PERCEPT — privacy-safe visual analysis workspace."""
from __future__ import annotations
import base64, html, sys, tempfile, time
from pathlib import Path
import cv2, numpy as np, streamlit as st
from PIL import Image, ImageDraw, ImageOps, ImageFilter
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT/'src'))
from pdc import PDCConfig, PDCPipeline
from pdc.ui_state import person_at_point, person_id_for_index, selection_is_valid

st.set_page_config(page_title='PERCEPT · Visual analysis', page_icon='◈', layout='wide', initial_sidebar_state='expanded')

def inject_styles():
    banner = ROOT / 'assets' / 'banner.png'
    banner_uri = 'data:image/png;base64,' + base64.b64encode(banner.read_bytes()).decode() if banner.exists() else ''
    styles = '''<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--ink:#edf7fb;--muted:#90a8b7;--line:rgba(151,198,218,.16);--cyan:#74ddff;--panel:rgba(14,28,38,.78)}
html,body,[class*="css"]{font-family:'Manrope',sans-serif}.stApp{background:radial-gradient(1100px 500px at 76% -4%,#153345 0%,transparent 64%),linear-gradient(145deg,#061016,#09151d 48%,#071016);color:var(--ink)}
.stApp:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.18;background-image:linear-gradient(rgba(255,255,255,.018) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.018) 1px,transparent 1px);background-size:52px 52px;mask-image:linear-gradient(#000,transparent 80%)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(9,23,32,.97),rgba(5,14,20,.97));border-right:1px solid var(--line)}[data-testid="stSidebar"]>div:first-child{padding:2rem 1.35rem}
.brand{display:flex;align-items:center;gap:11px;margin-bottom:2.2rem}.brand-mark{display:grid;place-items:center;width:34px;height:34px;border:1px solid #64d8fa;border-radius:11px;color:var(--cyan);box-shadow:0 0 24px #4dcdf12e;font:700 18px 'Space Grotesk'}.brand-name{font:700 18px 'Space Grotesk';letter-spacing:.12em}
.eyebrow,.section-label{color:var(--cyan);font:600 10px 'Space Grotesk';letter-spacing:.2em;text-transform:uppercase}.section-label{color:#7894a4;letter-spacing:.16em;margin:12px 0}
.hero{position:relative;overflow:hidden;min-height:270px;display:flex;align-items:flex-end;padding:32px 38px;margin:0 0 25px;border:1px solid var(--line);border-radius:24px;background-image:linear-gradient(90deg,rgba(5,15,21,.98),rgba(5,15,21,.76) 40%,rgba(5,15,21,.16)),linear-gradient(0deg,rgba(6,16,22,.55),transparent 65%),url('/app/static/assets/banner.png');background-size:cover;background-position:center 38%;box-shadow:0 24px 60px #0003}.hero:after{content:"";position:absolute;inset:auto 0 0;height:1px;background:linear-gradient(90deg,var(--cyan),transparent 62%)}.hero-copy{position:relative;z-index:1;max-width:620px}.hero h1{margin:8px 0 10px;font:600 clamp(2.2rem,4vw,4.3rem)/.98 'Space Grotesk';letter-spacing:-.055em}.hero p{margin:0;color:#c0d1d8;max-width:470px;font-size:15px;line-height:1.7;letter-spacing:.005em}
.source-card{margin:0 0 25px;padding:20px 22px;border:1px solid var(--line);border-radius:18px;background:linear-gradient(135deg,rgba(18,39,50,.72),rgba(9,20,27,.72));box-shadow:0 14px 38px #0000001c}.source-heading{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:10px}.source-title{color:#eefaff;font:600 17px 'Space Grotesk'}.source-copy{color:#8fa9b5;font-size:13px;line-height:1.55}.source-pulse{width:8px;height:8px;display:inline-block;border-radius:50%;background:var(--cyan);box-shadow:0 0 0 5px #74ddff1c,0 0 18px #74ddff}.stFileUploader,.stCameraInput{border:1px solid #74ddff29;border-radius:14px;background:#07182066}.stFileUploader section,.stCameraInput section{padding:10px 14px}
.media-frame{padding:14px;border:1px solid #97c6da24;border-radius:14px;background:radial-gradient(circle at 50% 35%,#183544,#09161d 68%);overflow:hidden}.media-frame [data-testid="stImage"]{display:flex;align-items:center;justify-content:center;min-height:420px}.media-frame [data-testid="stImage"] img{display:block;width:100%;height:420px;object-fit:contain;border-radius:9px}
.surface [data-testid="stImage"]{display:flex;align-items:center;justify-content:center;min-height:420px;padding:14px;border:1px solid #97c6da24;border-radius:14px;background:radial-gradient(circle at 50% 35%,#183544,#09161d 68%);overflow:hidden}.surface [data-testid="stImage"] img{display:block;width:100%;height:420px;object-fit:contain;border-radius:9px}
[data-testid="stSidebar"] [role="radiogroup"]{padding:10px 12px;border:1px solid #97c6da24;border-radius:14px;background:#0a1a23;margin:10px 0 24px;gap:6px}[data-testid="stSidebar"] [role="radiogroup"] label{display:flex;align-items:center;gap:8px;margin:0!important;padding:9px 10px;border-radius:9px;color:#a6bdc7!important;font-size:13px!important;transition:.2s}[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:#153340;color:#ecfaff!important}[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:linear-gradient(90deg,#164556,#102d39);color:#ecfaff!important;box-shadow:inset 2px 0 var(--cyan)}[data-testid="stSidebar"] [role="radiogroup"] label p{font-weight:600!important}
.surface{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:18px;box-shadow:0 14px 38px #00000024}.metric{padding:3px 0 7px 16px;border-left:1px solid #3b6170}.metric-label{color:#7894a4;font:600 10px 'Space Grotesk';letter-spacing:.13em;text-transform:uppercase}.metric-value{margin-top:5px;color:#f0f8fb;font:600 20px 'Space Grotesk'}
.person-card{position:relative;overflow:hidden;margin:0 0 12px;padding:14px;border:1px solid var(--line);border-radius:16px;background:linear-gradient(135deg,#142731e6,#0a161de0);transition:.2s}.person-card:hover{transform:translateY(-2px);border-color:#74ddff75}.person-card.selected{border-color:var(--cyan);background:linear-gradient(135deg,#143542f2,#0a1b24f2);box-shadow:0 0 0 1px #74ddff29,0 12px 30px #2baace1f}.person-card.selected:before{content:"";position:absolute;left:0;top:14px;bottom:14px;width:3px;border-radius:0 4px 4px 0;background:var(--cyan)}.person-name{color:#f2fbff;font:600 14px 'Space Grotesk';letter-spacing:.04em}.person-summary{color:#96b0bd;font-size:12px;margin-top:4px}.data-row{display:flex;justify-content:space-between;gap:8px;padding:8px 0;border-bottom:1px solid #97c6da1a;color:#a4bbc5;font-size:12px}.data-row b{color:#ecf8fb;font-weight:500;text-align:right}.empty-state{padding:52px 22px;border:1px dashed #97c6da40;border-radius:18px;text-align:center;color:var(--muted)}.empty-state strong{display:block;margin-bottom:7px;color:#dcebf0;font:600 18px 'Space Grotesk'}
.stButton>button{border:1px solid #74ddff47;border-radius:10px;background:#142e3ac7;color:#dff7ff;font-weight:600;min-height:38px;transition:.2s}.stButton>button:hover{border-color:var(--cyan);background:#173f4e;color:#fff}.stSlider label,.stRadio label{color:#b6cbd3!important}div[data-testid="stAlert"]{border-radius:12px}@media(max-width:800px){.hero{min-height:300px;padding:24px;border-radius:18px;background-position:64% center}.hero h1{font-size:2.5rem}.source-card{padding:16px}.media-frame{min-height:300px}.media-frame img{height:300px}.surface{padding:14px}.metric{padding-left:10px}}

/* Final symmetry and interaction pass */
html,body,[class*="css"],p,label,button,input,small{font-family:'Manrope',sans-serif!important}
p{line-height:1.65}.eyebrow,.section-label,.metric-label{font-family:'Manrope',sans-serif!important;font-weight:700}
.source-card{margin:0 0 18px;padding:20px 22px;border:1px solid var(--line);border-radius:18px;background:linear-gradient(145deg,rgba(16,38,49,.92),rgba(7,20,27,.94));box-shadow:0 16px 42px #0003}
.source-heading{display:grid;grid-template-columns:44px minmax(0,1fr);align-items:center;gap:14px;margin:0}.source-icon{display:grid;place-items:center;width:44px;height:44px;border:1px solid #74ddff39;border-radius:13px;background:linear-gradient(145deg,#164354,#0b2935);color:var(--cyan);font:700 18px 'Space Grotesk'}.source-text{min-width:0}.source-title{font:700 16px/1.25 'Manrope';letter-spacing:-.01em}.source-copy{margin-top:4px;color:#8fa9b5;font:500 12px/1.5 'Manrope'}
.upload-zone{margin:0 0 25px;padding:0}[data-testid="stFileUploader"],[data-testid="stCameraInput"]{padding:0!important;border:0!important;border-radius:18px!important;background:transparent!important}
[data-testid="stFileUploader"] section,[data-testid="stCameraInput"] section{min-height:112px;padding:18px!important;border:1px dashed #74ddff55!important;border-radius:18px!important;background:linear-gradient(145deg,rgba(10,31,41,.9),rgba(6,20,28,.94))!important;box-shadow:inset 0 1px rgba(255,255,255,.025);transition:border-color .18s,background .18s,transform .18s}
[data-testid="stFileUploader"] section:hover,[data-testid="stCameraInput"] section:hover{transform:translateY(-1px);border-color:var(--cyan)!important;background:linear-gradient(145deg,#0d2b37,#071f29)!important}
[data-testid="stFileUploader"] label,[data-testid="stCameraInput"] label{margin:0 0 8px!important;color:#c9dbe2!important;font:650 12px 'Manrope'!important}
[data-testid="stFileUploader"] small{color:#7f99a5!important;font:500 11px 'Manrope'!important}
[data-testid="stCameraInput"] video,[data-testid="stCameraInput"] img{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;border-radius:13px;background:#071219}
.media-frame,[data-testid="stImageCoordinates"]{background:#071219!important}.media-frame [data-testid="stImage"] img,.surface [data-testid="stImage"] img{width:100%!important;height:auto!important;aspect-ratio:16/9!important;object-fit:contain!important;background:#071219!important;border-radius:12px!important}
.quick-focus{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:0 0 8px}.quick-focus-title{color:#dcecf2;font:700 12px 'Manrope'}.quick-focus-copy{color:#79939f;font:500 11px 'Manrope'}
.person-card{min-height:70px;padding:15px;background:linear-gradient(145deg,#132b36,#091820);box-shadow:0 14px 34px #0003}.person-name{font:700 13px/1.3 'Manrope';letter-spacing:0}.person-summary{font:500 11px/1.45 'Manrope'}.data-row{font:550 11px/1.35 'Manrope'}.data-row b{font:650 11px/1.35 'Manrope'}
.person-photo [data-testid="stImage"] img{display:block;width:100%;aspect-ratio:4/3;object-fit:cover;border-radius:12px;background:#071219}
.stButton>button{font:700 11px 'Manrope';letter-spacing:.01em}.metric-value{font-size:19px}.empty-state{font:500 12px/1.6 'Manrope'}
@media(max-width:800px){.source-card{padding:16px}.source-heading{grid-template-columns:40px minmax(0,1fr);gap:12px}.source-icon{width:40px;height:40px}.source-meta{display:none}.source-title{font-size:15px}.source-copy{font-size:11px}.hero p{font-size:13px;line-height:1.65}.section-label{margin-top:20px}.metric-label{font-size:8px}.metric-value{font-size:16px}.quick-focus{align-items:flex-start;flex-direction:column;gap:3px}}
@media(max-width:480px){.source-card{border-radius:16px}.source-copy{max-width:28ch}.hero{padding:22px}.hero h1{letter-spacing:-.045em}.person-card{padding:13px}.data-row{padding:7px 0}}


/* Refined responsive blur presentation and professional uploader */
.upload-zone{margin:0 0 24px!important;padding:0!important}
.upload-zone [data-testid="stFileUploader"],.upload-zone [data-testid="stCameraInput"]{overflow:hidden;border:1px solid rgba(130,195,218,.22)!important;border-radius:16px!important;background:linear-gradient(145deg,rgba(13,31,41,.96),rgba(6,18,25,.98))!important;box-shadow:0 14px 34px rgba(0,0,0,.22),inset 0 1px rgba(255,255,255,.035)!important}
.upload-zone [data-testid="stFileUploader"]>label,.upload-zone [data-testid="stCameraInput"]>label{padding:13px 16px 0!important;margin:0!important;color:#dcecf2!important;font:700 11px/1.35 'Manrope'!important;letter-spacing:.01em}
.upload-zone [data-testid="stFileUploader"] section,.upload-zone [data-testid="stCameraInput"] section{display:flex!important;align-items:center!important;justify-content:center!important;gap:18px!important;min-height:92px!important;padding:16px 18px!important;border:0!important;border-radius:0!important;background:transparent!important;box-shadow:none!important;transition:background .18s ease!important}
.upload-zone [data-testid="stFileUploader"] section:hover,.upload-zone [data-testid="stCameraInput"] section:hover{transform:none!important;background:rgba(116,221,255,.035)!important}
.upload-zone [data-testid="stFileUploader"] section>div{gap:4px!important}.upload-zone [data-testid="stFileUploader"] section svg{color:#73ddff!important;filter:drop-shadow(0 0 10px rgba(115,221,255,.22))}
.upload-zone [data-testid="stFileUploader"] section button{min-width:126px!important;min-height:42px!important;padding:0 18px!important;border:1px solid rgba(115,221,255,.55)!important;border-radius:11px!important;background:linear-gradient(145deg,#1b5266,#123746)!important;color:#f2fbff!important;box-shadow:0 8px 20px rgba(0,0,0,.2),inset 0 1px rgba(255,255,255,.08)!important;font:750 11px 'Manrope'!important;letter-spacing:.015em!important}
.upload-zone [data-testid="stFileUploader"] section button:hover{border-color:#73ddff!important;background:linear-gradient(145deg,#21657c,#164757)!important;transform:translateY(-1px)!important;box-shadow:0 11px 24px rgba(24,154,194,.18)!important}
.upload-zone [data-testid="stFileUploaderFile"]{margin:0 14px 14px!important;padding:11px 12px!important;border:1px solid rgba(138,190,210,.16)!important;border-radius:11px!important;background:rgba(4,14,20,.62)!important;box-shadow:none!important}
.upload-zone [data-testid="stFileUploaderFile"] button{border:0!important;background:transparent!important;box-shadow:none!important;min-height:auto!important}
.upload-zone [data-testid="stFileUploader"] small{color:#7894a1!important;font:500 10px/1.45 'Manrope'!important}
.upload-zone [data-testid="stCameraInput"] video,.upload-zone [data-testid="stCameraInput"] img{max-height:min(62vh,620px)!important;aspect-ratio:16/9!important;object-fit:contain!important;border-radius:12px!important;background:radial-gradient(circle at center,#102a35,#061219 72%)!important}
.media-frame [data-testid="stImage"] img,.surface [data-testid="stImage"] img{background:radial-gradient(circle at center,#102a35,#061219 72%)!important}
@media(max-width:700px){.upload-zone [data-testid="stFileUploader"] section,.upload-zone [data-testid="stCameraInput"] section{flex-direction:column!important;gap:10px!important;min-height:118px!important;padding:17px 14px!important;text-align:center!important}.upload-zone [data-testid="stFileUploader"] section button{width:100%!important;max-width:240px!important}.upload-zone [data-testid="stFileUploader"]>label,.upload-zone [data-testid="stCameraInput"]>label{padding:12px 14px 0!important}.source-meta{display:none}.source-heading{grid-template-columns:42px minmax(0,1fr)!important}}


/* Premium upload component v3 */
.upload-zone{position:relative;margin:0 0 26px!important;padding:1px!important;border-radius:20px!important;background:linear-gradient(135deg,rgba(115,221,255,.34),rgba(115,221,255,.06) 38%,rgba(103,232,178,.16))!important;box-shadow:0 22px 55px rgba(0,0,0,.28)!important}
.upload-zone:before{content:"";position:absolute;inset:1px;border-radius:19px;pointer-events:none;background:radial-gradient(560px 150px at 15% 0%,rgba(115,221,255,.09),transparent 67%)}
.upload-zone [data-testid="stFileUploader"],.upload-zone [data-testid="stCameraInput"]{position:relative;z-index:1;overflow:hidden;margin:0!important;border:0!important;border-radius:19px!important;background:linear-gradient(145deg,rgba(12,31,41,.985),rgba(5,17,24,.99))!important;box-shadow:inset 0 1px rgba(255,255,255,.045)!important}
.upload-zone [data-testid="stFileUploader"]>label,.upload-zone [data-testid="stCameraInput"]>label{display:flex!important;align-items:center!important;min-height:39px!important;margin:0!important;padding:13px 18px 8px!important;border-bottom:1px solid rgba(143,199,220,.10)!important;color:#8da8b5!important;font:750 9px/1.3 'Manrope'!important;letter-spacing:.105em!important}
.upload-zone [data-testid="stFileUploader"] section{display:grid!important;grid-template-columns:minmax(0,1fr) auto!important;align-items:center!important;gap:24px!important;min-height:118px!important;padding:23px 24px!important;border:0!important;border-radius:0!important;background:transparent!important;box-shadow:none!important;transition:background .2s ease!important}
.upload-zone [data-testid="stFileUploader"] section:hover{transform:none!important;background:linear-gradient(90deg,rgba(115,221,255,.035),rgba(115,221,255,.012))!important}
.upload-zone [data-testid="stFileUploader"] section>div{min-width:0!important;gap:5px!important}
.upload-zone [data-testid="stFileUploader"] section svg{width:32px!important;height:32px!important;padding:7px!important;border:1px solid rgba(115,221,255,.25)!important;border-radius:11px!important;background:rgba(115,221,255,.075)!important;color:#73ddff!important;filter:drop-shadow(0 0 12px rgba(115,221,255,.16))!important}
.upload-zone [data-testid="stFileUploader"] section span,.upload-zone [data-testid="stFileUploader"] section p{color:#dcecf2!important;font:650 13px/1.45 'Manrope'!important}
.upload-zone [data-testid="stFileUploader"] section small{display:block!important;margin-top:3px!important;color:#728e9b!important;font:500 10px/1.55 'Manrope'!important}
.upload-zone [data-testid="stFileUploader"] section button{position:relative!important;min-width:154px!important;min-height:46px!important;padding:0 20px!important;border:1px solid rgba(115,221,255,.72)!important;border-radius:12px!important;background:linear-gradient(180deg,#1e5d72,#174658)!important;color:#f6fcff!important;box-shadow:0 10px 24px rgba(0,0,0,.25),inset 0 1px rgba(255,255,255,.14)!important;font:750 12px/1 'Manrope'!important;letter-spacing:.015em!important;transition:transform .17s ease,border-color .17s ease,box-shadow .17s ease,background .17s ease!important}
.upload-zone [data-testid="stFileUploader"] section button:hover{transform:translateY(-2px)!important;border-color:#a8efff!important;background:linear-gradient(180deg,#267189,#1a5366)!important;box-shadow:0 14px 30px rgba(24,163,204,.22),inset 0 1px rgba(255,255,255,.16)!important}
.upload-zone [data-testid="stFileUploader"] section button:active{transform:translateY(0)!important}
.upload-zone [data-testid="stFileUploaderFile"]{margin:0 18px 18px!important;padding:12px 14px!important;border:1px solid rgba(143,199,220,.16)!important;border-radius:13px!important;background:linear-gradient(145deg,rgba(6,19,27,.88),rgba(9,25,33,.88))!important;box-shadow:inset 0 1px rgba(255,255,255,.025)!important}
.upload-zone [data-testid="stFileUploaderFile"]>div{gap:10px!important}.upload-zone [data-testid="stFileUploaderFile"] button{min-width:auto!important;min-height:30px!important;padding:4px!important;border:0!important;background:transparent!important;box-shadow:none!important;color:#91abb7!important}.upload-zone [data-testid="stFileUploaderFile"] button:hover{transform:none!important;color:#fff!important}
.upload-zone [data-testid="stCameraInput"] section{padding:18px!important;border:0!important;background:transparent!important}.upload-zone [data-testid="stCameraInput"] button{border:1px solid rgba(115,221,255,.7)!important;border-radius:12px!important;background:linear-gradient(180deg,#1e5d72,#174658)!important;color:#f6fcff!important;box-shadow:0 10px 24px rgba(0,0,0,.24)!important;font:750 12px 'Manrope'!important}
@media(max-width:720px){.upload-zone{border-radius:17px!important}.upload-zone [data-testid="stFileUploader"],.upload-zone [data-testid="stCameraInput"]{border-radius:16px!important}.upload-zone [data-testid="stFileUploader"] section{grid-template-columns:1fr!important;gap:16px!important;min-height:158px!important;padding:20px 17px!important;text-align:center!important}.upload-zone [data-testid="stFileUploader"] section>div{justify-self:center!important}.upload-zone [data-testid="stFileUploader"] section button{justify-self:stretch!important;width:100%!important;min-width:0!important}.upload-zone [data-testid="stFileUploader"]>label,.upload-zone [data-testid="stCameraInput"]>label{justify-content:center!important;padding-inline:12px!important;text-align:center!important}.upload-zone [data-testid="stFileUploaderFile"]{margin:0 12px 12px!important}}

/* Direct widget styling: Streamlit widgets are siblings of markdown blocks. */
[data-testid="stFileUploader"],[data-testid="stCameraInput"]{overflow:hidden!important;border:1px solid rgba(130,195,218,.28)!important;border-radius:18px!important;background:linear-gradient(145deg,rgba(12,31,41,.98),rgba(5,17,24,.99))!important;box-shadow:0 18px 42px rgba(0,0,0,.24)!important}
[data-testid="stFileUploader"] section{display:grid!important;grid-template-columns:minmax(0,1fr) auto!important;align-items:center!important;gap:22px!important;min-height:124px!important;padding:22px 24px!important;border:0!important;background:linear-gradient(90deg,rgba(116,221,255,.025),transparent 70%)!important}
[data-testid="stFileUploader"] section button,[data-testid="stCameraInput"] button{min-width:152px!important;min-height:45px!important;border:1px solid rgba(115,221,255,.72)!important;border-radius:12px!important;background:linear-gradient(180deg,#1e5d72,#174658)!important;color:#f6fcff!important;font:700 12px 'Manrope'!important;box-shadow:0 10px 24px rgba(0,0,0,.25)!important;transition:.18s!important}
[data-testid="stFileUploader"] section button:hover,[data-testid="stCameraInput"] button:hover{transform:translateY(-2px)!important;border-color:#a8efff!important;background:linear-gradient(180deg,#267189,#1a5366)!important}
[data-testid="stCameraInput"] section{padding:18px!important;border:0!important;background:transparent!important}[data-testid="stCameraInput"] video,[data-testid="stCameraInput"] img{width:100%!important;max-height:520px!important;aspect-ratio:16/9!important;object-fit:contain!important;border-radius:12px;background:#071219}
@media(max-width:700px){[data-testid="stFileUploader"] section{grid-template-columns:1fr!important;min-height:158px!important;text-align:center!important}[data-testid="stFileUploader"] section button{width:100%!important}}

/* Integrated source action: the old session badge and detached upload zone are removed. */
.st-key-source_action_card{margin:0 0 24px!important;padding:18px 20px!important;border:1px solid rgba(151,198,218,.18)!important;border-radius:18px!important;background:linear-gradient(145deg,rgba(16,38,49,.94),rgba(7,20,27,.96))!important;box-shadow:0 16px 42px rgba(0,0,0,.28),inset 0 1px rgba(255,255,255,.025)!important}
.st-key-source_action_card [data-testid="stHorizontalBlock"]{align-items:center!important;gap:18px!important}.st-key-source_action_card [data-testid="column"]{min-width:0!important}
.source-inline{display:grid;grid-template-columns:44px minmax(0,1fr);align-items:center;gap:14px;min-height:48px}.source-inline .source-icon{width:44px;height:44px}.source-inline .source-title{font:700 16px/1.25 'Manrope';letter-spacing:-.01em}.source-inline .source-copy{margin-top:4px;font:500 12px/1.5 'Manrope'}
.st-key-source_action_card [data-testid="stFileUploader"]{overflow:visible!important;margin:0!important;padding:0!important;border:0!important;border-radius:0!important;background:transparent!important;box-shadow:none!important}.st-key-source_action_card [data-testid="stFileUploader"]>label{display:none!important}.st-key-source_action_card [data-testid="stFileUploader"] section{display:flex!important;align-items:center!important;justify-content:flex-end!important;min-height:46px!important;margin:0!important;padding:0!important;border:0!important;background:transparent!important;box-shadow:none!important}.st-key-source_action_card [data-testid="stFileUploader"] section>div,.st-key-source_action_card [data-testid="stFileUploader"] section svg,.st-key-source_action_card [data-testid="stFileUploader"] section small{display:none!important}.st-key-source_action_card [data-testid="stFileUploaderFile"]{display:none!important}
.st-key-source_action_card [data-testid="stFileUploader"] section button,.st-key-source_action_card [data-testid="stPopover"]>button{width:100%!important;min-width:148px!important;min-height:44px!important;margin:0!important;padding:0 17px!important;border:1px solid rgba(115,221,255,.7)!important;border-radius:12px!important;background:linear-gradient(180deg,#1e5d72,#174658)!important;color:#f6fcff!important;font:700 12px/1 'Manrope'!important;letter-spacing:.01em!important;box-shadow:0 10px 24px rgba(0,0,0,.24),inset 0 1px rgba(255,255,255,.12)!important;transition:transform .17s,border-color .17s,background .17s,box-shadow .17s!important}.st-key-source_action_card [data-testid="stFileUploader"] section button:hover,.st-key-source_action_card [data-testid="stPopover"]>button:hover{transform:translateY(-1px)!important;border-color:#a8efff!important;background:linear-gradient(180deg,#267189,#1a5366)!important;box-shadow:0 13px 27px rgba(24,163,204,.18)!important}
.selected-source{margin:9px 0 0;color:#75909c;font:600 10px/1.4 'Manrope';text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.selected-source b{color:#b9d0d9;font-weight:700}
@media(max-width:760px){.st-key-source_action_card{padding:16px!important}.st-key-source_action_card [data-testid="stHorizontalBlock"]{gap:13px!important}.source-inline{grid-template-columns:40px minmax(0,1fr);gap:12px}.source-inline .source-icon{width:40px;height:40px}.source-inline .source-title{font-size:14px}.source-inline .source-copy{font-size:10.5px}.st-key-source_action_card [data-testid="stFileUploader"] section button,.st-key-source_action_card [data-testid="stPopover"]>button{min-width:122px!important;padding:0 12px!important;font-size:11px!important}.selected-source{font-size:9px}}
@media(max-width:520px){.st-key-source_action_card [data-testid="stHorizontalBlock"]{display:flex!important;flex-direction:column!important;align-items:stretch!important}.st-key-source_action_card [data-testid="column"]{width:100%!important;flex:1 1 auto!important}.st-key-source_action_card [data-testid="stFileUploader"] section{justify-content:stretch!important}.st-key-source_action_card [data-testid="stFileUploader"] section button,.st-key-source_action_card [data-testid="stPopover"]>button{width:100%!important}.selected-source{text-align:left;margin-top:7px}.source-inline .source-copy{max-width:none}}


/* Prediction reset and upload-card spacing refinement */
.source-inline .source-copy{margin-top:5px!important;margin-bottom:7px!important;padding-bottom:1px!important}
.st-key-source_action_card [data-testid="stFileUploader"]{display:flex!important;flex-direction:column!important;justify-content:center!important;min-height:52px!important}
.st-key-source_action_card [data-testid="stFileUploader"] section{min-height:44px!important}
.selected-source{display:flex!important;align-items:center!important;justify-content:flex-end!important;gap:5px!important;min-height:18px!important;margin:4px 1px 0!important;padding:0!important;color:#75909c!important;font:600 10px/1.2 'Manrope'!important;transform:none!important}.selected-source b{color:#c7dce4!important;font-weight:700!important}.selected-source:before{content:"";display:block;width:5px;height:5px;border-radius:50%;background:#67e8b2;box-shadow:0 0 0 3px rgba(103,232,178,.09)}
.st-key-run_another_wrap{margin:30px auto 8px!important;padding:0!important;max-width:420px!important}.st-key-run_another_wrap [data-testid="stButton"]{margin:0!important}.st-key-run_another_wrap button{width:100%!important;min-height:50px!important;border:1px solid rgba(115,221,255,.7)!important;border-radius:14px!important;background:linear-gradient(180deg,#1d596e,#143f50)!important;color:#f5fcff!important;font:750 12px/1 'Manrope'!important;letter-spacing:.025em!important;box-shadow:0 14px 30px rgba(0,0,0,.26),inset 0 1px rgba(255,255,255,.13)!important;transition:transform .18s,border-color .18s,background .18s,box-shadow .18s!important}.st-key-run_another_wrap button:hover{transform:translateY(-2px)!important;border-color:#a8efff!important;background:linear-gradient(180deg,#26748c,#194d60)!important;box-shadow:0 17px 34px rgba(27,169,211,.2),inset 0 1px rgba(255,255,255,.16)!important}.st-key-run_another_wrap button:active{transform:translateY(0)!important}.reset-note{margin:8px 0 0;text-align:center;color:#718c98;font:500 10px/1.45 'Manrope'}
@media(max-width:760px){.source-inline .source-copy{margin-bottom:9px!important}.selected-source{justify-content:flex-end!important;margin-top:3px!important;font-size:9px!important}.st-key-run_another_wrap{max-width:none!important;margin:25px 0 6px!important}}
@media(max-width:520px){.source-inline .source-copy{margin-bottom:11px!important}.selected-source{justify-content:flex-start!important;margin-top:5px!important}.st-key-run_another_wrap button{min-height:48px!important;border-radius:13px!important}}


/* Source-card alignment correction */
.st-key-source_action_card [data-testid="stHorizontalBlock"]{align-items:center!important}
.st-key-source_action_card [data-testid="column"]{display:flex!important;flex-direction:column!important;justify-content:center!important}
.source-inline{display:grid!important;grid-template-columns:44px minmax(0,1fr)!important;align-items:center!important;column-gap:14px!important;min-height:54px!important;width:100%!important}
.source-inline .source-icon{display:grid!important;place-items:center!important;align-self:center!important;width:44px!important;height:44px!important;min-width:44px!important;margin:0!important;padding:0!important;line-height:1!important;transform:none!important}
.source-inline .source-text{display:flex!important;flex-direction:column!important;justify-content:center!important;align-self:stretch!important;min-width:0!important;margin:0!important;padding:0!important}
.source-inline .source-title{margin:0 0 4px!important;padding:0!important;font:700 16px/1.2 'Manrope'!important}
.source-inline .source-copy{margin:0!important;padding:0!important;font:500 12px/1.45 'Manrope'!important}
@media(max-width:760px){.source-inline{grid-template-columns:42px minmax(0,1fr)!important;column-gap:12px!important;min-height:50px!important}.source-inline .source-icon{width:42px!important;height:42px!important;min-width:42px!important}.source-inline .source-title{font-size:15px!important}.source-inline .source-copy{margin:0!important;font-size:11px!important}}

</style>'''.replace("/app/static/assets/banner.png", banner_uri)
    st.markdown(styles,unsafe_allow_html=True)

@st.cache_resource(show_spinner='Preparing analysis…')
def load_pipeline(confidence, pose_confidence, max_people):
    config=PDCConfig.load(); config.model.detection_confidence=confidence; config.model.pose_keypoint_visibility=pose_confidence; return PDCPipeline(config)
def display_id(result): return result.person_id or person_id_for_index(result.person_index)
def profile_label(result,pipeline):
    return 'Unavailable' if result.analysis is None else pipeline.config.visualization.gender_labels.get(result.analysis.category_index,'Unclassified')
def selected_frame(frame,selected_id):
    image=frame.annotated_image.copy()
    selected=next((r for r in frame.analysis_results if display_id(r)==selected_id),None) if selected_id else None
    if selected:
        draw=ImageDraw.Draw(image,'RGBA'); x1,y1,x2,y2=selected.person.box; draw.rounded_rectangle((x1,y1,x2,y2),radius=9,outline=(116,221,255,255),width=4)
    return image
def crop_for_card(image_bgr,result):
    ih,iw=image_bgr.shape[:2]; x1,y1,x2,y2=map(int,result.person.box)
    bw,bh=max(1,x2-x1),max(1,y2-y1)
    x1=max(0,x1-int(bw*.16)); x2=min(iw,x2+int(bw*.16))
    y1=max(0,y1-int(bh*.12)); y2=min(ih,y2+int(bh*.08))
    crop=image_bgr[y1:y2,x1:x2]
    if not crop.size: return Image.new('RGB',(720,540),'#071219')
    rgb=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
    return ImageOps.fit(rgb,(720,540),method=Image.Resampling.LANCZOS,centering=(.5,.42))
def normalize_media(image, width=1280, height=720):
    """Create responsive cinematic blur bars while preserving the full source image."""
    image=ImageOps.exif_transpose(image).convert('RGB')
    target=(width,height)
    # A cover-sized copy becomes the responsive ambient background. Blur strength
    # scales slightly with source/target mismatch so extreme portraits stay calm.
    source_ratio=image.width/max(1,image.height)
    target_ratio=width/max(1,height)
    mismatch=min(1.0,abs(np.log(max(source_ratio,1e-6)/target_ratio)))
    blur_radius=int(22+18*mismatch)
    background=ImageOps.fit(image,target,method=Image.Resampling.LANCZOS,centering=(.5,.5))
    background=background.filter(ImageFilter.GaussianBlur(blur_radius))
    tint=Image.new('RGB',target,'#061219')
    background=Image.blend(background,tint,.44+.12*mismatch)
    fitted=ImageOps.contain(image,target,method=Image.Resampling.LANCZOS)
    x=(width-fitted.width)//2; y=(height-fitted.height)//2
    # A restrained shadow separates the sharp image from the ambient blur bars.
    shadow=Image.new('RGBA',target,(0,0,0,0))
    shadow_draw=ImageDraw.Draw(shadow,'RGBA')
    shadow_draw.rounded_rectangle((max(0,x-8),max(0,y-8),min(width,x+fitted.width+8),min(height,y+fitted.height+8)),radius=14,fill=(0,0,0,88))
    canvas=Image.alpha_composite(background.convert('RGBA'),shadow).convert('RGB')
    canvas.paste(fitted,(x,y))
    return canvas
def render_card(result,source_bgr,pipeline,selected_id):
    person_id=display_id(result); active=person_id==selected_id; meta=result.metadata or {}; age='Unavailable' if result.analysis is None else f'{result.analysis.estimated_age:.0f} years'; visibility=meta.get('body_visibility','Unavailable').replace('_',' ').title(); orientation=meta.get('camera_orientation','Unavailable').replace('_',' ').title()
    st.markdown(f'<div class="person-card {"selected" if active else ""}"><div class="person-name">{person_id}</div><div class="person-summary">{orientation} · {visibility}</div></div>',unsafe_allow_html=True)
    photo,details=st.columns([.82,1.18],gap='small')
    with photo:
        st.markdown('<div class="person-photo">',unsafe_allow_html=True); st.image(crop_for_card(source_bgr,result),use_container_width=True); st.markdown('</div>',unsafe_allow_html=True)
    with details:
        for label,value in (('Estimated age',age),('Profile',profile_label(result,pipeline)),('Confidence',f'{result.person.confidence:.0%}')): st.markdown(f'<div class="data-row"><span>{label}</span><b>{value}</b></div>',unsafe_allow_html=True)
        if st.button('Selected' if active else 'Focus',key=f'select-{person_id}',use_container_width=True): st.session_state.selected_person=person_id; st.rerun()

def reset_prediction():
    """Return to a fresh uploader by rotating widget keys before the next render."""
    st.session_state.upload_generation=st.session_state.get('upload_generation',0)+1
    for key in ('_analysis_cache','selected_person'):
        st.session_state.pop(key,None)

def main():
    inject_styles()
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-mark">◈</div><div class="brand-name">PERCEPT</div></div>',unsafe_allow_html=True); st.markdown('<div class="eyebrow">Workspace</div>',unsafe_allow_html=True)
        mode=st.radio('Source',['Image','Live Camera','Video'],label_visibility='collapsed'); st.divider(); st.markdown('<div class="eyebrow">Analysis controls</div>',unsafe_allow_html=True)
        confidence=st.slider('Detection sensitivity',.05,.95,.25,.05); pose_confidence=st.slider('Detail sensitivity',.05,.95,.50,.05); max_people=st.slider('People shown',1,20,10); frame_sampling=st.slider('Video sampling',1,30,1) if mode=='Video' else 1; st.divider(); st.caption('Your media is used only for this analysis session.')
    st.markdown('<section class="hero"><div class="hero-copy"><div class="eyebrow">Visual intelligence workspace</div><h1>See the scene<br>with clarity.</h1><p>Bring an image, a live frame, or a video into focus. Explore the people, context, and visual signals that matter.</p></div></section>',unsafe_allow_html=True)
    source_title = {'Image':'Add an image','Live Camera':'Capture a live frame','Video':'Add a video'}[mode]
    source_copy = {'Image':'Upload one JPG, PNG, or WebP image. Maximum file size: 5 MB.','Live Camera':'Use your camera for a live, one-frame capture.','Video':'Choose a clip and select the sampling pace in the sidebar.'}[mode]
    source_icon={'Image':'▧','Live Camera':'◉','Video':'▷'}[mode]
    uploaded=None
    upload_generation=st.session_state.get('upload_generation',0)
    with st.container(key='source_action_card'):
        source_info,source_action=st.columns([4.7,1.3],gap='medium',vertical_alignment='center')
        with source_info:
            st.markdown(f'<div class="source-inline"><div class="source-icon">{source_icon}</div><div class="source-text"><div class="source-title">{source_title}</div><div class="source-copy">{source_copy}</div></div></div>',unsafe_allow_html=True)
        with source_action:
            if mode=='Image':
                uploaded=st.file_uploader('Upload image',type=['jpg','jpeg','png','webp'],key=f'image-source-{upload_generation}',label_visibility='collapsed',help='Choose one JPG, JPEG, PNG, or WebP image up to 5 MB.')
            elif mode=='Video':
                uploaded=st.file_uploader('Upload video',type=['mp4','mov','avi','mkv'],key=f'video-source-{upload_generation}',label_visibility='collapsed',help='Choose one MP4, MOV, AVI, or MKV video.')
            else:
                if hasattr(st,'popover'):
                    with st.popover('Open camera',use_container_width=True):
                        uploaded=st.camera_input('Capture or retake frame',key=f'camera-source-{upload_generation}',label_visibility='collapsed')
                else:
                    uploaded=st.camera_input('Capture or retake frame',key=f'camera-source-{upload_generation}',label_visibility='collapsed')
            if uploaded is not None:
                source_name=html.escape(str(getattr(uploaded,'name','Camera capture')),quote=True)
                st.markdown(f'<div class="selected-source"><b>Selected</b> · {source_name}</div>',unsafe_allow_html=True)
    if uploaded is None: st.markdown('<div class="empty-state"><strong>Your workspace is ready</strong>Choose a source above to begin a visual analysis.</div>',unsafe_allow_html=True); return
    media_bytes=uploaded.getvalue()
    if not media_bytes:
        st.warning('The selected source is empty. Please choose another file or retake the camera frame.')
        return
    if mode=='Image' and len(media_bytes)>5*1024*1024:
        actual_mb=len(media_bytes)/(1024*1024)
        st.error(f'Image exceeds the 5 MB limit ({actual_mb:.2f} MB selected). Please compress or resize it and upload again.',icon='⚠️')
        st.caption('Accepted formats: JPG, JPEG, PNG and WebP. Maximum image size: 5 MB.')
        return
    import hashlib
    analysis_key=(mode,hashlib.sha1(media_bytes).hexdigest(),confidence,pose_confidence,max_people,frame_sampling)
    cached=st.session_state.get('_analysis_cache')
    if cached and cached.get('key')==analysis_key:
        frame,image,elapsed,pipeline=cached['frame'],cached['image'],cached['elapsed'],load_pipeline(confidence,pose_confidence,max_people)
    elif mode=='Video':
        suffix=Path(uploaded.name).suffix or '.mp4'
        with tempfile.NamedTemporaryFile(suffix=suffix) as temporary_video:
            temporary_video.write(uploaded.getvalue()); temporary_video.flush(); capture=cv2.VideoCapture(temporary_video.name)
            if not capture.isOpened(): st.error('This video could not be opened. Please try another file.'); return
            try:
                pipeline=load_pipeline(confidence,pose_confidence,max_people); total_frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1; progress=st.progress(0,text='Reviewing video…'); frame_number=0; latest=processed=None
                while True:
                    ok,candidate=capture.read()
                    if not ok: break
                    if frame_number%frame_sampling==0: processed=pipeline.process_image(candidate); processed.analysis_results=processed.analysis_results[:max_people]; latest=candidate
                    frame_number+=1; progress.progress(min(frame_number/total_frames,1.0),text='Reviewing video…')
                progress.empty()
            except Exception: st.error('We couldn’t complete this analysis. Please try again with another source.'); return
            finally: capture.release()
        if processed is None or latest is None: st.warning('No readable moments were found in this video.'); return
        frame,image,elapsed=processed,latest,processed.timing_ms.get('total',0)
    elif not (cached and cached.get('key')==analysis_key):
        raw=np.frombuffer(media_bytes,np.uint8); image=cv2.imdecode(raw,cv2.IMREAD_COLOR)
        if image is None: st.error('This file could not be read. Please choose a JPEG, PNG, or WebP image.'); return
        try:
            pipeline=load_pipeline(confidence,pose_confidence,max_people); started=time.perf_counter(); frame=pipeline.process_image(image); frame.analysis_results=frame.analysis_results[:max_people]; elapsed=(time.perf_counter()-started)*1000
        except Exception: st.error('We couldn’t complete this analysis. Please try again with another source.'); return
    if not (cached and cached.get('key')==analysis_key):
        st.session_state['_analysis_cache']={'key':analysis_key,'frame':frame,'image':image,'elapsed':elapsed}
    if not selection_is_valid(frame.analysis_results,st.session_state.get('selected_person')): st.session_state.selected_person=display_id(frame.analysis_results[0]) if frame.analysis_results else None
    selected_id=st.session_state.get('selected_person'); fps=1000/frame.timing_ms['total'] if frame.timing_ms.get('total') else None; st.markdown('<div class="section-label">Session overview</div>',unsafe_allow_html=True)
    metrics=st.columns(4)
    for col,label,value in zip(metrics,('System status','People found','Response time','Live rate'),('Ready',str(frame.person_count),f'{frame.timing_ms.get("total",elapsed):.0f} ms',f'{fps:.1f} fps' if fps else '—')):
        with col: st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-label">Visual field</div>',unsafe_allow_html=True)
    if frame.analysis_results:
        st.markdown('<div class="quick-focus"><div class="quick-focus-title">Quick focus</div><div class="quick-focus-copy">Switch subjects without repeating analysis</div></div>',unsafe_allow_html=True)
        focus_cols=st.columns(min(len(frame.analysis_results),6),gap='small')
        for focus_index,focus_result in enumerate(frame.analysis_results):
            focus_id=display_id(focus_result)
            with focus_cols[focus_index%len(focus_cols)]:
                if st.button(('● ' if focus_id==selected_id else '')+f'{focus_index+1:02d}',key=f'quick-{focus_id}',use_container_width=True,disabled=focus_id==selected_id):
                    st.session_state.selected_person=focus_id; st.rerun()
    display_image=selected_frame(frame,selected_id)
    source_width, source_height = display_image.size
    display_image = normalize_media(display_image)
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
        click=streamlit_image_coordinates(display_image,key=f'frame-{analysis_key[1]}')
        if click:
            scale = min(1280 / source_width, 720 / source_height)
            source_x = (click['x'] - (1280 - source_width * scale) / 2) / scale
            source_y = (click['y'] - (720 - source_height * scale) / 2) / scale
            clicked_person = person_at_point(frame.analysis_results, source_x, source_y)
            if clicked_person: st.session_state.selected_person=clicked_person; st.rerun()
    except ImportError: st.image(display_image,use_container_width=True)
    st.markdown('<div class="section-label">People & insights</div>',unsafe_allow_html=True)
    if not frame.analysis_results:
        st.markdown('<div class="empty-state"><strong>No people in focus</strong>Try a clearer frame or adjust sensitivity.</div>',unsafe_allow_html=True)
    else:
        person_count=len(frame.analysis_results)
        if person_count==1:
            _,only_col,_=st.columns([1,1.35,1],gap='medium'); card_columns=[only_col]
        else:
            card_columns=st.columns(2 if person_count==2 else 3,gap='medium')
        for index,result in enumerate(frame.analysis_results):
            with card_columns[index%len(card_columns)]: render_card(result,image,pipeline,selected_id)
    with st.container(key='run_another_wrap'):
        st.button('↻  Run another prediction',key='run-another-prediction',use_container_width=True,on_click=reset_prediction)
        st.markdown('<div class="reset-note">Clears the current source and returns to a fresh upload state.</div>',unsafe_allow_html=True)
if __name__=='__main__': main()

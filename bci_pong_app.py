"""
BCI Simulation — Brain-Computer Interface Pipeline
Real PhysioNet EEG → Signal Processing → ML Classification
"""

import os, time, warnings
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy import signal as scipy_signal
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
DURATION = 2.0
N_CH = 64
IMAGERY_RUNS = [4, 8, 12]  # Motor imagery: left/right fist

CH_NAMES = [
    "Fc5","Fc3","Fc1","Fcz","Fc2","Fc4","Fc6",
    "C5","C3","C1","Cz","C2","C4","C6",
    "Cp5","Cp3","Cp1","Cpz","Cp2","Cp4","Cp6",
    "Fp1","Fpz","Fp2","Af7","Af3","Afz","Af4","Af8",
    "F7","F5","F3","F1","Fz","F2","F4","F6","F8",
    "Ft7","Ft8","T7","T8","T9","T10","Tp7","Tp8",
    "P7","P5","P3","P1","Pz","P2","P4","P6","P8",
    "Po7","Po3","Poz","Po4","Po8","O1","Oz","O2","Iz",
]
C3, C4 = CH_NAMES.index("C3"), CH_NAMES.index("C4")
CP3, CP4 = CH_NAMES.index("Cp3"), CH_NAMES.index("Cp4")


# ── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_physionet(subject: int):
    """Download real EEG motor imagery data from PhysioNet via MNE."""
    import mne
    from mne.datasets import eegbci
    from mne.io import concatenate_raws, read_raw_edf

    mne.set_log_level("ERROR")

    fnames = eegbci.load_data(subject, IMAGERY_RUNS, update_path=False)
    raws = [read_raw_edf(f, preload=True) for f in fnames]
    for raw in raws:
        raw.rename_channels(lambda x: x.rstrip('.'))

    raw = concatenate_raws(raws)
    fs = int(raw.info["sfreq"])
    events, event_id = mne.events_from_annotations(raw)

    t1_code = event_id.get("T1")
    t2_code = event_id.get("T2")
    if t1_code is None or t2_code is None:
        return None

    n_samp = int(DURATION * fs)
    data = raw.get_data()
    n_eeg = min(data.shape[0], 64)

    left, right = [], []
    for ev in events:
        onset, code = ev[0], ev[2]
        if onset + n_samp > data.shape[1]:
            continue
        trial = data[:n_eeg, onset:onset + n_samp] * 1e6  # to uV
        if trial.shape[0] < 64:
            trial = np.vstack([trial, np.zeros((64 - trial.shape[0], n_samp))])
        if code == t1_code:
            left.append(trial)
        elif code == t2_code:
            right.append(trial)

    if len(left) < 3 or len(right) < 3:
        return None
    return np.array(left), np.array(right), fs


def extract_features(trial, fs):
    feats = []
    n = trial.shape[1]
    for ch in [C3, C4, CP3, CP4]:
        f, psd = scipy_signal.welch(trial[ch], fs=fs,
                                     nperseg=min(256, n), noverlap=min(128, n // 2))
        feats.append(np.mean(psd[(f >= 8) & (f <= 12)]))
        feats.append(np.mean(psd[(f >= 13) & (f <= 30)]))
    return np.array(feats)


# ── Session State ────────────────────────────────────────────────────────────

def init():
    d = dict(
        loaded=False, trials_l=None, trials_r=None, fs=160,
        subject=1, n_left=0, n_right=0,
        clf=None, trained=False, acc=0.0,
        side=None, eeg=None, ran=False,
        pred=None, conf=0.0,
        idx_l=0, idx_r=0,
        history=[],
    )
    for k, v in d.items():
        if k not in st.session_state:
            st.session_state[k] = v


def train_clf():
    if st.session_state.trained or not st.session_state.loaded:
        return
    s = st.session_state
    X, y = [], []
    for t in s.trials_l:
        X.append(extract_features(t, s.fs)); y.append(0)
    for t in s.trials_r:
        X.append(extract_features(t, s.fs)); y.append(1)
    X, y = np.array(X), np.array(y)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    folds = min(5, min(np.sum(y == 0), np.sum(y == 1)))
    s.acc = float(np.mean(cross_val_score(pipe, X, y, cv=max(2, folds)))) if folds >= 2 else 0
    pipe.fit(X, y)
    s.clf = pipe
    s.trained = True


def classify(trial, fs):
    f = extract_features(trial, fs).reshape(1, -1)
    p = st.session_state.clf.predict(f)[0]
    prob = st.session_state.clf.predict_proba(f)[0]
    return ("LEFT" if p == 0 else "RIGHT"), float(prob[p])


# ── Plotly ───────────────────────────────────────────────────────────────────

DK = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
          font=dict(color="#94a3b8", size=11, family="Inter, sans-serif"),
          margin=dict(l=40, r=20, t=36, b=36))
GR = dict(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.08)")


def viz_psd(data, side, fs):
    chs = [(C3, "C3 left", "#38bdf8"), (C4, "C4 right", "#4ade80"),
           (CP3, "CP3 left", "#f59e0b"), (CP4, "CP4 right", "#f87171")]
    fig = go.Figure()
    n = data.shape[1]
    for ci, nm, col in chs:
        f, p = scipy_signal.welch(data[ci], fs=fs, nperseg=min(256, n), noverlap=min(128, n // 2))
        fig.add_trace(go.Scatter(x=f, y=p, mode="lines", name=nm, line=dict(color=col, width=1.5)))
    fig.add_vrect(x0=8, x1=12, fillcolor="rgba(56,189,248,0.06)", line_width=0,
                  annotation_text="mu", annotation_position="top left",
                  annotation_font=dict(color="#38bdf8", size=10))
    fig.add_vrect(x0=13, x1=30, fillcolor="rgba(74,222,128,0.06)", line_width=0,
                  annotation_text="beta", annotation_position="top left",
                  annotation_font=dict(color="#4ade80", size=10))
    fig.update_layout(**DK, xaxis=dict(**GR, title="Hz", range=[0, 50]),
                      yaxis=dict(**GR, title="Power"), height=320,
                      legend=dict(bgcolor="rgba(0,0,0,0)", x=1, xanchor="right", y=1))
    return fig


def viz_spectrogram(data, side, fs):
    ch = C4 if side == "left" else C3
    nm = "C4 right" if side == "left" else "C3 left"
    n = data.shape[1]
    nperseg = min(64, n // 2)
    noverlap = nperseg - max(1, nperseg // 8)
    f, t, S = scipy_signal.spectrogram(data[ch], fs=fs, nperseg=nperseg,
                                        noverlap=noverlap, window="hann")
    mask = f <= 50
    fig = go.Figure(go.Heatmap(
        z=10 * np.log10(S[mask] + 1e-12), x=t, y=f[mask],
        colorscale=[[0, "#020617"], [0.3, "#1e3a5f"], [0.6, "#0e7490"],
                     [0.85, "#38bdf8"], [1, "#f0f9ff"]],
        colorbar=dict(title=dict(text="dB", font=dict(color="#94a3b8")),
                      tickfont=dict(color="#94a3b8"), thickness=10, len=0.6),
    ))
    fig.update_layout(**DK, xaxis=dict(**GR, title="Time (s)"),
                      yaxis=dict(**GR, title="Hz"), height=320)
    return fig, nm


def viz_covariance(data):
    n = data.shape[1]
    cov = data @ data.T / n
    ticks = list(range(0, 64, 8))
    fig = go.Figure(go.Heatmap(
        z=cov,
        colorscale=[[0, "#020617"], [0.35, "#1e3a5f"], [0.65, "#0e7490"],
                     [0.85, "#38bdf8"], [1, "#f0f9ff"]],
        colorbar=dict(title=dict(text="Cov", font=dict(color="#94a3b8")),
                      tickfont=dict(color="#94a3b8"), thickness=10, len=0.6),
    ))
    for ch in [C3, C4, CP3, CP4]:
        fig.add_shape(type="rect", x0=ch - .5, x1=ch + .5, y0=ch - .5, y1=ch + .5,
                      line=dict(color="#f87171", width=1.5))
    fig.update_layout(**DK,
                      xaxis=dict(**GR, title="Ch", tickvals=ticks, ticktext=[CH_NAMES[i] for i in ticks]),
                      yaxis=dict(**GR, title="Ch", tickvals=ticks, ticktext=[CH_NAMES[i] for i in ticks],
                                 autorange="reversed"), height=380)
    return fig, cov


def viz_decode(label, conf, side):
    """Simple left/right hemisphere decode indicator."""
    left_col = "#38bdf8" if label == "LEFT" else "#1e293b"
    right_col = "#4ade80" if label == "RIGHT" else "#1e293b"
    left_op = 1.0 if label == "LEFT" else 0.2
    right_op = 1.0 if label == "RIGHT" else 0.2

    fig = go.Figure()

    # Left hemisphere
    fig.add_shape(type="circle", x0=0.5, y0=0.2, x1=4.5, y1=4.8,
                  fillcolor=left_col, line=dict(color="#38bdf8", width=2),
                  opacity=left_op)
    # Right hemisphere
    fig.add_shape(type="circle", x0=5.5, y0=0.2, x1=9.5, y1=4.8,
                  fillcolor=right_col, line=dict(color="#4ade80", width=2),
                  opacity=right_op)
    # Center line
    fig.add_shape(type="line", x0=5, y0=0, x1=5, y1=5,
                  line=dict(color="#334155", width=1, dash="dot"))

    # Labels
    fig.add_annotation(x=2.5, y=2.5, text="LEFT", showarrow=False,
                       font=dict(color="#f8fafc" if label == "LEFT" else "#475569",
                                 size=20, family="monospace"))
    fig.add_annotation(x=7.5, y=2.5, text="RIGHT", showarrow=False,
                       font=dict(color="#f8fafc" if label == "RIGHT" else "#475569",
                                 size=20, family="monospace"))

    # Confidence bar
    bar_x = 2.5 if label == "LEFT" else 7.5
    bar_col = "#38bdf8" if label == "LEFT" else "#4ade80"
    fig.add_annotation(x=5, y=-0.5, text=f"{conf:.0%} confidence", showarrow=False,
                       font=dict(color=bar_col, size=14, family="monospace"))

    # Imagined label
    fig.add_annotation(x=5, y=5.5, text=f"Imagined: {side.upper()} hand", showarrow=False,
                       font=dict(color="#94a3b8", size=12))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[-0.5, 10.5], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-1.2, 6.2], showgrid=False, zeroline=False, visible=False,
                   scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=0, b=0), height=240, showlegend=False,
    )
    return fig


# ── CSS ──────────────────────────────────────────────────────────────────────

CSS = """
<style>
:root { --bg: #020617; --surface: #0f172a; --border: #1e293b; --text: #cbd5e1; --accent: #38bdf8; }
.stApp { background: var(--bg); color: var(--text); }
section[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
h1,h2,h3,h4 { color: #f1f5f9 !important; font-weight: 500 !important; }
[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'JetBrains Mono', monospace; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #64748b; border-bottom: 2px solid transparent; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; background: transparent !important; }
.decode-badge { display:inline-block; padding:8px 20px; border-radius:6px; font-family:'JetBrains Mono',monospace; font-size:1em; margin:4px 0; }
.decode-left { background:rgba(56,189,248,0.12); border:1px solid rgba(56,189,248,0.3); color:#38bdf8; }
.decode-right { background:rgba(74,222,128,0.12); border:1px solid rgba(74,222,128,0.3); color:#4ade80; }
.explain { color:#94a3b8; font-size:0.88em; line-height:1.6; }
.math-card { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:16px 20px; margin:8px 0; }
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="BCI Simulation", page_icon="🧠", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    init()

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🧠 BCI Simulation")
        st.caption("PhysioNet EEG Motor Imagery")
        st.markdown("---")

        if not st.session_state.loaded:
            st.markdown("**Load real EEG data**")
            subject = st.number_input("Subject (1–109)", 1, 109, 1, key="subj_input")
            st.caption("Each subject is a different person who had EEG recorded while imagining hand movements.")
            if st.button("Download from PhysioNet", key="dl"):
                with st.spinner("Downloading EEG recordings..."):
                    result = load_physionet(subject)
                if result is not None:
                    s = st.session_state
                    s.trials_l, s.trials_r, s.fs = result
                    s.n_left, s.n_right = len(s.trials_l), len(s.trials_r)
                    s.loaded = True
                    s.subject = subject
                    s.idx_l = s.idx_r = 0
                    s.trained = False
                    st.rerun()
                else:
                    st.error("Download failed. Check your connection and try again.")
        else:
            s = st.session_state
            st.markdown(f"**Subject {s.subject}** loaded")
            st.caption(f"{s.n_left} left + {s.n_right} right trials · {s.fs} Hz · 64 ch")

            st.markdown("---")
            st.markdown("**Run a trial:**")
            c1, c2 = st.columns(2)
            if c1.button("✋ Left", key="bl"):
                s.side = "left"
                s.eeg = s.trials_l[s.idx_l % s.n_left]
                s.idx_l += 1
                s.ran = True
            if c2.button("🤚 Right", key="br"):
                s.side = "right"
                s.eeg = s.trials_r[s.idx_r % s.n_right]
                s.idx_r += 1
                s.ran = True

            st.markdown("---")
            if st.button("Load different subject"):
                for k in ["loaded", "trained", "ran", "trials_l", "trials_r",
                           "clf", "history", "pred", "conf", "eeg", "side"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

            if s.trained:
                st.caption(f"Classifier accuracy: **{s.acc:.0%}**")

    # ── Not loaded yet ───────────────────────────────────────────────────
    if not st.session_state.loaded:
        st.markdown("## 🧠 BCI Simulation")
        st.markdown(
            '<p class="explain" style="font-size:1em; max-width:600px;">'
            'This app uses <strong>real EEG brain recordings</strong> from the '
            '<a href="https://physionet.org/content/eegmmidb/1.0.0/" style="color:#38bdf8">'
            'PhysioNet Motor Movement/Imagery Dataset</a>. '
            '109 volunteers had 64 electrodes placed on their scalp and were asked to '
            '<em>imagine</em> moving their left or right hand. '
            'Select a subject in the sidebar to download their brain data and start decoding.'
            '</p>', unsafe_allow_html=True)
        return

    # ── Train classifier ─────────────────────────────────────────────────
    if not st.session_state.trained:
        with st.spinner("Training classifier on real brain data..."):
            train_clf()

    # ── Header ───────────────────────────────────────────────────────────
    s = st.session_state
    st.markdown("## 🧠 BCI Simulation")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Subject", f"#{s.subject}")
    m2.metric("Trials", f"{s.n_left + s.n_right}")
    m3.metric("Accuracy", f"{s.acc:.0%}")
    m4.metric("Rate", f"{s.fs} Hz")

    # ── Decode result ────────────────────────────────────────────────────
    if s.ran and s.eeg is not None:
        data, side, fs = s.eeg, s.side, s.fs
        label, conf = classify(data, fs)
        s.pred, s.conf = label, conf
        s.history.append(dict(side=side, pred=label, conf=conf))

        st.markdown("---")

        # Decode indicator
        col_viz, col_info = st.columns([3, 2])
        with col_viz:
            st.plotly_chart(viz_decode(label, conf, side), width="stretch")
        with col_info:
            cls = "decode-left" if label == "LEFT" else "decode-right"
            correct = (side == "left" and label == "LEFT") or (side == "right" and label == "RIGHT")
            icon = "✓" if correct else "✗"
            st.markdown(
                f'<div class="decode-badge {cls}" style="font-size:1.1em; padding:12px 24px;">'
                f'{icon} Decoded: <strong>{label}</strong> ({conf:.0%})</div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<p class="explain" style="margin-top:12px;">'
                f'The classifier examined alpha and beta band power across motor cortex channels '
                f'and determined this brain recording most likely corresponds to '
                f'<strong>{label.lower()} hand</strong> motor imagery.</p>',
                unsafe_allow_html=True)

        # ── History ──────────────────────────────────────────────────
        if len(s.history) > 1:
            total = len(s.history)
            correct_n = sum(1 for h in s.history
                           if (h["side"] == "left" and h["pred"] == "LEFT")
                           or (h["side"] == "right" and h["pred"] == "RIGHT"))
            st.caption(f"Session: {correct_n}/{total} correct ({correct_n/total:.0%})")

        # ── Pipeline ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Signal Processing")

        t1, t2, t3 = st.tabs(["Frequency", "Time-Frequency", "Spatial"])

        with t1:
            ca, cb = st.columns([3, 2])
            with ca:
                st.plotly_chart(viz_psd(data, side, fs), width="stretch")
            with cb:

                st.markdown("**Fourier Transform**")
                st.latex(r"X(k) = \sum_{t=0}^{T-1} x(t)\, e^{-j\frac{2\pi}{T} k t}")
                sup = "C4/CP4 (right)" if side == "left" else "C3/CP3 (left)"
                st.markdown(
                    f'<p class="explain">Splits the signal into frequencies. '
                    f'Mu (8–12 Hz) and Beta (13–30 Hz) carry motor intent. '
                    f'During {side} imagery, power drops on {sup} — '
                    f'this is Event-Related Desynchronization.</p>', unsafe_allow_html=True)


        with t2:
            ca, cb = st.columns([3, 2])
            with ca:
                fig_s, ch_nm = viz_spectrogram(data, side, fs)
                st.plotly_chart(fig_s, width="stretch")
            with cb:

                st.markdown("**Wavelet / STFT**")
                st.latex(r"W(a,b) = \frac{1}{\sqrt{|a|}} \int x(t)\, \psi\!\left(\frac{t-b}{a}\right) dt")
                st.markdown(
                    f'<p class="explain">Shows channel {ch_nm} (contralateral). '
                    f'Reveals when frequency changes happen over the 2-second window.</p>',
                    unsafe_allow_html=True)


        with t3:
            ca, cb = st.columns([3, 2])
            with ca:
                fig_c, cov = viz_covariance(data)
                st.plotly_chart(fig_c, width="stretch")
            with cb:

                st.markdown("**Spatial Covariance**")
                st.latex(r"\Sigma = \frac{1}{T}\, X\, X^{\mathsf{T}}")
                c3v, c4v = cov[C3, C3], cov[C4, C4]
                st.markdown(
                    f'<p class="explain">64×64 matrix of channel correlations. '
                    f'C3: {c3v:.1f} / C4: {c4v:.1f} — '
                    f'lower value = desynchronized hemisphere.</p>',
                    unsafe_allow_html=True)


        # Features
        with st.expander("Feature vector"):
            feats = extract_features(data, fs)
            names = ["C3 α", "C3 β", "C4 α", "C4 β", "CP3 α", "CP3 β", "CP4 α", "CP4 β"]
            colors = ["#38bdf8", "#0ea5e9", "#4ade80", "#22c55e",
                       "#f59e0b", "#d97706", "#f87171", "#ef4444"]
            fig_f = go.Figure(go.Bar(x=names, y=feats, marker_color=colors))
            fig_f.update_layout(**DK, height=240, xaxis=dict(**GR), yaxis=dict(**GR, title="Power"))
            st.plotly_chart(fig_f, width="stretch")
            st.markdown(
                '<p class="explain">These 8 values are what the Logistic Regression '
                'classifier sees: alpha + beta power from 4 motor channels.</p>',
                unsafe_allow_html=True)

    else:
        st.markdown("---")
        st.markdown(
            '<div style="text-align:center; padding:50px; color:#475569;">'
            '<p style="font-size:1.1em;">Click <strong>Left</strong> or '
            '<strong>Right</strong> in the sidebar to decode a trial</p>'
            '<p>Each click pulls the next real EEG recording from this subject</p>'
            '</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

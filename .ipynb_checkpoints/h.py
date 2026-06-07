import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #e53e3e, #c53030);
        color: white;
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 25px;
        text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p  { margin: 5px 0 0; font-size: 1rem; opacity: 0.9; }

    .result-positive {
        background: linear-gradient(135deg, #fff5f5, #fed7d7);
        border: 2px solid #e53e3e;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        margin: 20px 0;
    }
    .result-negative {
        background: linear-gradient(135deg, #f0fff4, #c6f6d5);
        border: 2px solid #38a169;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        margin: 20px 0;
    }
    .result-positive h2 { color: #c53030; }
    .result-negative h2 { color: #276749; }

    .prob-box {
        background: white;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 8px 4px;
    }
    .section-header {
        background: #e53e3e;
        color: white;
        padding: 8px 16px;
        border-radius: 8px;
        margin: 15px 0 10px;
        font-weight: 600;
    }
    .tip-box {
        background: #ebf8ff;
        border-left: 4px solid #3182ce;
        padding: 12px 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 0.9rem;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    div[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    div[data-testid="stSidebar"] .stRadio > label { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Load Model & Data ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("Heart_Disease.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("heart_dataset.csv")

model = load_model()
df    = load_data()

FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak']

# ─── Sidebar Navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ❤️ Heart Disease App")
    st.markdown("---")
    page = st.radio(
        "📌 Section Chuniye:",
        ["🩺 Heart Disease Prediction", "📊 Data Graphs & Analysis"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**ℹ️ Model Info**")
    st.markdown(f"- **Algorithm:** Random Forest")
    st.markdown(f"- **Features Used:** {len(FEATURES)}")
    st.markdown(f"- **Dataset Records:** {len(df)}")
    st.markdown("---")
    st.markdown("**⚠️ Disclaimer**")
    st.markdown("*Ye sirf educational tool hai. Doctor se zaroor milein.*", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🩺 Heart Disease Prediction":

    st.markdown("""
    <div class="main-header">
        <h1>❤️ Heart Disease Prediction System</h1>
        <p>Apni health details bharein aur AI se heart disease ka risk jaanein</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📝 Apni Health Information Bharein")
    st.info("💡 Niche diye gaye fields mein apni jaankari daalein, phir **Predict** button dabaein.")

    # ── Input Section ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="section-header">👤 Personal Info</div>', unsafe_allow_html=True)

        age = st.number_input(
            "🎂 Aayu (Umar in Years)",
            min_value=18, max_value=100, value=45,
            help="Apni aayu darj karein (18–100)"
        )

        sex = st.selectbox(
            "⚧ Ling (Gender)",
            options=[0, 1],
            format_func=lambda x: "👩 Mahila (Female)" if x == 0 else "👨 Purush (Male)",
            help="Apna gender chuniye"
        )

        cp = st.selectbox(
            "💔 Seene ka Dard (Chest Pain Type)",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 – Koi dard nahi (No Pain / Asymptomatic)",
                1: "1 – Halka typical dard (Mild Angina)",
                2: "2 – Atypical dard (Atypical Angina)",
                3: "3 – Seene mein tej dard (Non-anginal Pain)"
            }[x],
            help="Seene ke dard ki prakar chuniye"
        )

    with col2:
        st.markdown('<div class="section-header">🩸 Blood Reports</div>', unsafe_allow_html=True)

        trestbps = st.number_input(
            "🩺 Aaram ki Stithi mein BP (Resting Blood Pressure - mmHg)",
            min_value=80, max_value=220, value=120,
            help="Normal: 90–120 mmHg"
        )

        chol = st.number_input(
            "🧪 Cholesterol ka Sthar (mg/dl)",
            min_value=100, max_value=600, value=200,
            help="Normal: <200 mg/dl. High: >240 mg/dl"
        )

        fbs = st.selectbox(
            "🍬 Bhojan se pehle Blood Sugar (Fasting Blood Sugar)",
            options=[0, 1],
            format_func=lambda x: (
                "✅ Normal (120 mg/dl se kum)" if x == 0
                else "⚠️ Zyada hai (120 mg/dl se adhik)"
            ),
            help="Khaane se pehle blood sugar 120 mg/dl se zyada hai ya kam?"
        )

    with col3:
        st.markdown('<div class="section-header">🫀 Heart Parameters</div>', unsafe_allow_html=True)

        thalach = st.number_input(
            "💓 Maximum Heart Rate (Exercise ke dauran)",
            min_value=60, max_value=220, value=150,
            help="Exercise ke waqt sabse zyada heart rate. Normal: 150–190 bpm"
        )

        exang = st.selectbox(
            "🏃 Exercise se Seene mein Dard?",
            options=[0, 1],
            format_func=lambda x: (
                "✅ Nahi, koi dard nahi" if x == 0
                else "⚠️ Haan, dard hota hai (Exercise-induced Angina)"
            ),
            help="Kya exercise karte waqt seene mein dard hota hai?"
        )

        oldpeak = st.number_input(
            "📉 ST Depression (Exercise vs Aaram mein)",
            min_value=0.0, max_value=10.0, value=1.0, step=0.1,
            help="ECG mein ST segment ka depression. Normal: 0–1"
        )

    st.markdown("---")

    # ── Predict Button ───────────────────────────────────────────────────────
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_clicked = st.button("🔍 Heart Disease ka Risk Jaanein", use_container_width=True, type="primary")

    if predict_clicked:
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, thalach, exang, oldpeak]])

        prediction   = model.predict(input_data)[0]
        probability  = model.predict_proba(input_data)[0]
        prob_disease = probability[1] * 100
        prob_healthy = probability[0] * 100

        st.markdown("---")
        st.markdown("## 🔬 Prediction Result")

        if prediction == 1:
            st.markdown(f"""
            <div class="result-positive">
                <h2>⚠️ Heart Disease ka Khatara Hai!</h2>
                <p style="font-size:1.1rem; color:#742a2a;">
                    AI model ke anusar, diye gaye symptoms ke aadhar par <strong>Heart Disease hone ki
                    sambhavna {prob_disease:.1f}%</strong> hai.<br>
                    Kripya turant kisi heart specialist (Cardiologist) se milein.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                <h2>✅ Heart Swasth Lag Raha Hai!</h2>
                <p style="font-size:1.1rem; color:#22543d;">
                    AI model ke anusar, <strong>Heart Disease hone ki sambhavna sirf {prob_disease:.1f}%</strong> hai.
                    Aapka dil swasth dikh raha hai!<br>
                    Phir bhi niyamit checkup zaroori hai.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Probability Gauge
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            fig_gauge, ax = plt.subplots(figsize=(6, 3))
            fig_gauge.patch.set_facecolor('#f8fafc')
            colors = ['#38a169', '#e53e3e']
            bars = ax.barh(['Heart Disease Risk'], [prob_disease], color='#e53e3e', height=0.4, label=f'Risk: {prob_disease:.1f}%')
            ax.barh(['Heart Disease Risk'], [100 - prob_disease], left=prob_disease, color='#38a169', height=0.4)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Probability (%)')
            ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
            ax.text(prob_disease / 2, 0, f"{prob_disease:.1f}%", ha='center', va='center', fontweight='bold', color='white', fontsize=13)
            ax.text(prob_disease + (100 - prob_disease) / 2, 0, f"{prob_healthy:.1f}%", ha='center', va='center', fontweight='bold', color='white', fontsize=13)
            red_patch  = mpatches.Patch(color='#e53e3e', label=f'Bimari ka Darr ({prob_disease:.1f}%)')
            green_patch = mpatches.Patch(color='#38a169', label=f'Swasth ({prob_healthy:.1f}%)')
            ax.legend(handles=[red_patch, green_patch], loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=2)
            ax.set_title('Risk Probability', fontweight='bold', pad=10)
            st.pyplot(fig_gauge, use_container_width=True)
            plt.close()

        # Input Summary
        st.markdown("### 📋 Aapke Diye Gaye Values ka Summary")
        summary = {
            "Parameter": ["Aayu", "Ling", "Seene ka Dard", "Aaram ka BP", "Cholesterol",
                          "Blood Sugar", "Max Heart Rate", "Exercise Angina", "ST Depression"],
            "Aapki Value": [
                f"{age} saal",
                "Mahila (Female)" if sex == 0 else "Purush (Male)",
                {0:"Asymptomatic", 1:"Mild Angina", 2:"Atypical Angina", 3:"Non-anginal Pain"}[cp],
                f"{trestbps} mmHg",
                f"{chol} mg/dl",
                "Normal (<120)" if fbs == 0 else "High (>120)",
                f"{thalach} bpm",
                "Nahi" if exang == 0 else "Haan",
                f"{oldpeak}"
            ],
            "Normal Range": [
                "18–100", "—", "0 (Asymptomatic)",
                "90–120 mmHg", "<200 mg/dl",
                "0 (Normal)", "150–190 bpm", "0 (Nahi)", "0–1"
            ]
        }
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="tip-box">
        💡 <strong>Doctor se zaroor milein!</strong> Ye AI tool sirf ek anumaan deta hai.
        Sahi diagnosis ke liye ECG, Blood Test aur Doctor ki salah zaroori hai.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.info("📊 **Graphs dekhne ke liye** → Sidebar mein **'Data Graphs & Analysis'** section par click karein!")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: GRAPHS
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("""
    <div class="main-header">
        <h1>📊 Heart Disease Data Analysis</h1>
        <p>Dataset ke aadhar par heart disease ke trends aur patterns</p>
    </div>
    """, unsafe_allow_html=True)

    # Rename for display
    df_display = df.copy()
    df_display['Disease'] = df_display['target'].map({1: 'Heart Disease', 0: 'Swasth (Healthy)'})
    df_display['Gender']  = df_display['sex'].map({1: 'Purush (Male)', 0: 'Mahila (Female)'})
    df_display['Seene ka Dard'] = df_display['cp'].map({
        0: 'Asymptomatic', 1: 'Mild Angina', 2: 'Atypical Angina', 3: 'Non-anginal'
    })
    df_display['Exercise Angina'] = df_display['exang'].map({0: 'Nahi', 1: 'Haan'})

    COLORS = {'Heart Disease': '#e53e3e', 'Swasth (Healthy)': '#38a169'}

    st.markdown("### 📈 Graphs & Analysis Chuniye")
    graph_choice = st.selectbox("Graph Chuniye:", [
        "1️⃣  Dataset Overview – Heart Disease Distribution",
        "2️⃣  Aayu aur Heart Disease ka Sambandh",
        "3️⃣  Gender-wise Heart Disease",
        "4️⃣  Seene ka Dard – Chest Pain Analysis",
        "5️⃣  Cholesterol aur Heart Disease",
        "6️⃣  Feature Importance (Model)",
        "7️⃣  Correlation Heatmap",
    ])

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#f8fafc')

    # ── Graph 1: Disease Distribution ────────────────────────────────────────
    if "1️⃣" in graph_choice:
        counts = df_display['Disease'].value_counts()
        wedge_props = dict(width=0.5, edgecolor='white', linewidth=3)
        ax.pie(counts, labels=counts.index,
               autopct='%1.1f%%', startangle=90,
               colors=['#e53e3e', '#38a169'],
               wedgeprops=wedge_props, textprops={'fontsize': 13})
        ax.set_title("Dataset mein Heart Disease ka Vitaran", fontsize=15, fontweight='bold', pad=15)
        total = len(df_display)
        ax.text(0, -1.35, f"Kul Patients: {total}  |  Heart Disease: {counts.get('Heart Disease',0)}  |  Swasth: {counts.get('Swasth (Healthy)',0)}",
                ha='center', fontsize=11, color='gray')

    # ── Graph 2: Age Distribution ─────────────────────────────────────────────
    elif "2️⃣" in graph_choice:
        for label, grp in df_display.groupby('Disease'):
            ax.hist(grp['age'], bins=15, alpha=0.7, label=label,
                    color=COLORS[label], edgecolor='white')
        ax.set_xlabel("Aayu (Years)", fontsize=12)
        ax.set_ylabel("Patients ki Sankhya", fontsize=12)
        ax.set_title("Aayu ke Hisaab se Heart Disease Distribution", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.axvline(df_display[df_display['target']==1]['age'].mean(), color='#e53e3e',
                   linestyle='--', alpha=0.8, label='HD Avg Age')
        avg_age = df_display[df_display['target']==1]['age'].mean()
        ax.text(avg_age+1, ax.get_ylim()[1]*0.85, f"HD Avg: {avg_age:.0f}yr",
                color='#e53e3e', fontsize=10)

    # ── Graph 3: Gender-wise ──────────────────────────────────────────────────
    elif "3️⃣" in graph_choice:
        gender_ct = df_display.groupby(['Gender', 'Disease']).size().unstack(fill_value=0)
        gender_ct.plot(kind='bar', ax=ax, color=['#38a169', '#e53e3e'],
                       edgecolor='white', width=0.6)
        ax.set_xlabel("Ling (Gender)", fontsize=12)
        ax.set_ylabel("Patients ki Sankhya", fontsize=12)
        ax.set_title("Gender ke Hisaab se Heart Disease", fontsize=14, fontweight='bold')
        ax.legend(title="Sthiti", fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=12)
        for container in ax.containers:
            ax.bar_label(container, fontsize=10, padding=3)

    # ── Graph 4: Chest Pain ───────────────────────────────────────────────────
    elif "4️⃣" in graph_choice:
        cp_ct = df_display.groupby(['Seene ka Dard', 'Disease']).size().unstack(fill_value=0)
        cp_ct.plot(kind='bar', ax=ax, color=['#38a169', '#e53e3e'],
                   edgecolor='white', width=0.6)
        ax.set_xlabel("Chest Pain Type", fontsize=12)
        ax.set_ylabel("Patients ki Sankhya", fontsize=12)
        ax.set_title("Seene ke Dard ka Prakar aur Heart Disease", fontsize=14, fontweight='bold')
        ax.legend(title="Sthiti", fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right', fontsize=10)
        for container in ax.containers:
            ax.bar_label(container, fontsize=10, padding=3)

    # ── Graph 5: Cholesterol ──────────────────────────────────────────────────
    elif "5️⃣" in graph_choice:
        for label, grp in df_display.groupby('Disease'):
            ax.scatter(grp['age'], grp['chol'], alpha=0.5, label=label,
                       color=COLORS[label], s=50)
        ax.set_xlabel("Aayu (Years)", fontsize=12)
        ax.set_ylabel("Cholesterol (mg/dl)", fontsize=12)
        ax.set_title("Aayu aur Cholesterol ke Saath Heart Disease", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.axhline(200, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label='Normal Limit (200)')
        ax.axhline(240, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='High Limit (240)')
        ax.text(ax.get_xlim()[1], 202, '← Normal', fontsize=9, color='orange')
        ax.text(ax.get_xlim()[1], 242, '← High', fontsize=9, color='red')

    # ── Graph 6: Feature Importance ───────────────────────────────────────────
    elif "6️⃣" in graph_choice:
        feat_labels = [
            "Aayu (Age)", "Ling (Sex)", "Seene ka Dard (CP)",
            "Aaram ka BP", "Cholesterol", "Blood Sugar (FBS)",
            "Max Heart Rate", "Exercise Angina", "ST Depression"
        ]
        importances = model.feature_importances_
        sorted_idx  = np.argsort(importances)[::-1]
        colors_bar  = ['#e53e3e' if i < 3 else '#4a90e2' for i in range(len(importances))]
        bars = ax.barh([feat_labels[i] for i in sorted_idx[::-1]],
                       importances[sorted_idx[::-1]], color=colors_bar[::-1],
                       edgecolor='white')
        ax.set_xlabel("Mahatva (Importance Score)", fontsize=12)
        ax.set_title("Heart Disease ke liye Sabse Zaroori Factors", fontsize=14, fontweight='bold')
        for bar, val in zip(bars, importances[sorted_idx[::-1]]):
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)

    # ── Graph 7: Correlation Heatmap ──────────────────────────────────────────
    elif "7️⃣" in graph_choice:
        plt.close()
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor('#f8fafc')
        rename_map = {
            'age':'Aayu', 'sex':'Ling', 'cp':'Seene Dard',
            'trestbps':'Aaram BP', 'chol':'Cholesterol',
            'fbs':'Blood Sugar', 'thalach':'Max HR',
            'exang':'Exer Angina', 'oldpeak':'ST Dep', 'target':'Heart Disease'
        }
        corr_cols = ['age','sex','cp','trestbps','chol','fbs','thalach','exang','oldpeak','target']
        corr_data = df[corr_cols].rename(columns=rename_map)
        mask = np.triu(np.ones_like(corr_data.corr(), dtype=bool))
        sns.heatmap(corr_data.corr(), mask=mask, annot=True, fmt='.2f',
                    cmap='RdYlGn', ax=ax, linewidths=0.5,
                    cbar_kws={"shrink": 0.8}, annot_kws={"size": 9})
        ax.set_title("Har Feature ka Heart Disease se Sambandh (Correlation)", fontsize=13, fontweight='bold', pad=15)
        plt.xticks(rotation=30, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)

    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Quick Stats
    st.markdown("---")
    st.markdown("### 📌 Quick Statistics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Kul Patients", len(df))
    with c2:
        st.metric("Heart Disease", df['target'].sum(), f"{df['target'].mean()*100:.1f}%")
    with c3:
        st.metric("Average Aayu", f"{df['age'].mean():.1f} saal")
    with c4:
        st.metric("Avg Cholesterol", f"{df['chol'].mean():.0f} mg/dl")
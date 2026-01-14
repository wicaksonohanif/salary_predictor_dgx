import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# CSS Halaman
st.markdown("""
<style>
.stApp {
    background: linear-gradient(360deg, #98cadb, #d7ebee);
}

div.stButton > button {
    font-size: 18px;
    padding: 10px 20px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Banner
st.image("./assets/banner_linreg.png")

# Sidebar
st.sidebar.markdown("""
<h1 style="font-size:25px; font-weight:800;">
    📊 Distribusi Data
</h1>
<p style="font-size:16px; font-weight:400;">
    Berikut adalah visualisasi distribusi data pengalaman kerja, jam kerja per minggu, dan gaji tahunan dari dataset yang digunakan untuk membangun model prediksi.
</p><br>
""", unsafe_allow_html=True)


df_sidebar = pd.read_csv("./dataset/salary_dataset_modified.csv")

fig_pengalaman = go.Figure()
fig_pengalaman.add_trace(go.Histogram(
    x=df_sidebar["Pengalaman"],
    nbinsx=20,
    marker_color="#4d879d"
))
fig_pengalaman.update_layout(
    title="Distribusi Pengalaman (Tahun)",
    xaxis=dict(
        title="Pengalaman (Tahun)"
    ),
    yaxis=dict(
        title="Jml. Karyawan"
    ),
    height=250,
    margin=dict(l=10, r=10, t=40, b=30)
)

st.sidebar.plotly_chart(fig_pengalaman, use_container_width=True)

fig_jamkerja = go.Figure()
fig_jamkerja.add_trace(go.Histogram(
    x=df_sidebar["JamKerjaPerMinggu"],
    nbinsx=20,
    marker_color="#98cadb"
))
fig_jamkerja.update_layout(
    title="Distribusi Jam Kerja / Minggu",
    xaxis=dict(
        title="Jam Kerja / Minggu"
    ),
    yaxis=dict(
        title="Jml. Karyawan"
    ),
    height=250,
    margin=dict(l=10, r=10, t=40, b=30)
)

st.sidebar.plotly_chart(fig_jamkerja, use_container_width=True)

fig_gaji = go.Figure()
fig_gaji.add_trace(go.Histogram(
    x=df_sidebar["Gaji"],
    nbinsx=20,
    marker_color="#39b986"
))
fig_gaji.update_layout(
    title="Distribusi Gaji Tahunan ($)",
    xaxis=dict(
        title="Gaji Tahunan ($)"
    ),
    yaxis=dict(
        title="Jml. Karyawan"
    ),
    height=250,
    margin=dict(l=10, r=10, t=40, b=30)
)

st.sidebar.plotly_chart(fig_gaji, use_container_width=True)

# Main Menu
st.markdown(
    """
    <h1 style="color:black; font-weight:800; font-size:40px;">
        💵 Aplikasi Prediksi Gaji Tahunan Karyawan
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <p style="color:black; font-weight:400; font-size:18px;">
        Masukkan pengalaman kerja dan jam kerja per minggu untuk memprediksi gaji karyawan.
    </p>
    """,
    unsafe_allow_html=True
)

# Load model dan scaler
lr_model = joblib.load("./models/model_regresi_gaji.joblib")
scaler = joblib.load("./models/robust_scaler.joblib")

# Input pengguna
st.markdown(
    "<p style='font-weight:800; font-size:16px; margin-bottom: -100px; color:black;'>Pengalaman Kerja (tahun)</p>",
    unsafe_allow_html=True
)
pengalaman = st.number_input(
    "",
    min_value=0,
    step=1,
    key="input_pengalaman"
)
st.markdown(
    "<p style='font-weight:800; font-size:16px; margin-bottom: -100px; color:black;'>Jam Kerja per Minggu</p>",
    unsafe_allow_html=True
)
jam_kerja = st.number_input(
    "",
    min_value=0,
    step=1,
    key="input_jam_kerja"
)

st.markdown("""
<style>
div.stButton > button {
    background-color: #39b986;
    color: white;
    font-size: 18px;
    padding: 10px 20px;
    border-radius: 10px;
    border: none;
}

div.stButton > button:hover {
    background-color: #349b72;
    color: white;
}
</style>
""", unsafe_allow_html=True)
prediksi = st.button("Prediksi Gaji")


# Prediksi Gaji
if prediksi:
    if not (1 <= pengalaman <= 11):
        st.markdown("""
            <div style="
                background-color: #e74c3c;
                padding: 20px;
                border-radius: 18px;
                text-align: center;
                margin-top: 25px;
            ">
                <div style="
                    color: white;
                    font-size: 18px; 
                    font-weight: 400">
                    ❌ Input Jam Kerja Tidak Valid<br>
                </div>
                <div style="
                    color: white;
                    font-size: 18px; 
                    font-weight: 600">
                    Pengalaman kerja harus berada pada rentang 1–11 tahun.
                </div>
            </div>
        """, unsafe_allow_html=True)

    elif not (34 <= jam_kerja <= 56):
        st.markdown("""
            <div style="
                background-color: #e74c3c;
                padding: 20px;
                border-radius: 18px;
                text-align: center;
                margin-top: 25px;
            ">
                <div style="
                    color: white;
                    font-size: 18px; 
                    font-weight: 400">
                    ❌ Input Jam Kerja Tidak Valid<br>
                </div>
                <div style="
                    color: white;
                    font-size: 18px; 
                    font-weight: 600">
                    Jam kerja per minggu harus berada pada rentang 34–56 jam.
                </div>
            </div>
        """, unsafe_allow_html=True)

    else:
        input_data = pd.DataFrame({
            "Pengalaman": [pengalaman],
            "JamKerjaPerMinggu": [jam_kerja]
        })

        input_scaled = scaler.transform(input_data)
        hasil = lr_model.predict(input_scaled)

        st.markdown(f"""
            <div style="
                    background-color: #39b986; 
                    padding: 20px; 
                    border-radius: 18px; 
                    text-align: center; 
                    margin-top: 25px;">
                <div style="
                    color: white; 
                    font-size: 18px; 
                    font-weight: 400;">
                    💰 Perkiraan Gaji:<br>
                </div>
                <div style="
                    color: white; 
                    font-size: 25px; 
                    font-weight: bold;">
                    ${hasil[0]:,.0f} / tahun
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin:25px;'></div>", unsafe_allow_html=True)

        gaji_min = 37732
        gaji_max = 122392
        gaji_prediksi = hasil[0]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=[gaji_min, gaji_max],
            y=[0, 0],
            mode="lines",
            line=dict(width=10, color="#98cadb"),
            name="Rentang Gaji"
        ))

        fig.add_trace(go.Scatter(
            x=[gaji_prediksi],
            y=[0],
            mode="markers",
            marker=dict(size=25, color="#39b986", symbol="diamond", line=dict(color="white", width=2)),
            name="Gaji Prediksi",
            hovertemplate=
            "<b>Gaji Prediksi</b><br>$%{x:,.0f}<extra></extra>"
        ))

        fig.update_layout(
            title=dict(
            text="Visualisasi Gaji Prediksi",
            font=dict(size=18),
            x=0.5,           
            xanchor="center"    
            ),
            xaxis=dict(
                title="Gaji Tahunan ($)",
                range=[gaji_min - 5000, gaji_max + 5000]
            ),
            yaxis=dict(
                visible=False
            ),
            height=200,
            margin=dict(l=10, r=10, t=50, b=30),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown(
    """
    <hr style="
        border: none;
        height: 1px;
        background-color: grey;
        margin-top: 40px;
        margin-bottom: 20px;
    ">

    <div style="text-align:center; font-size:12px; color:grey;">
        Copyright © 2026 by Pengelola MK Praktikum Unggulan (Praktikum DGX), Universitas Gunadarma
        <br>
        <a href="https://www.praktikum-hpc.gunadarma.ac.id/" target="_blank" style="color:grey;">
            https://www.praktikum-hpc.gunadarma.ac.id/
        </a>
        <br>
        <a href="https://www.hpc-hub.gunadarma.ac.id/" target="_blank" style="color:grey;">
            https://www.hpc-hub.gunadarma.ac.id/
        </a>
    </div>
    """,
    unsafe_allow_html=True
)



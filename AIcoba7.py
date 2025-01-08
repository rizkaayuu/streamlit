import streamlit as st
import pandas as pd
import joblib
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Kelas DecisionTree dan RandomForest
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, data, target, features, depth=0):
        if depth == self.max_depth or len(np.unique(data[target])) == 1:
            most_common_label = Counter(data[target]).most_common(1)[0][0]
            return most_common_label

        best_feature = features[0]
        tree = {best_feature: {}}

        for value in np.unique(data[best_feature]):
            subset = data[data[best_feature] == value]
            if subset.empty:
                most_common_label = Counter(data[target]).most_common(1)[0][0]
                tree[best_feature][value] = most_common_label
            else:
                tree[best_feature][value] = self.fit(subset, target, [f for f in features if f != best_feature], depth + 1)

        self.tree = tree
        return tree

    def predict_row(self, row, tree):
        if not isinstance(tree, dict):
            return tree

        feature = next(iter(tree))
        if row[feature] in tree[feature]:
            return self.predict_row(row, tree[feature][row[feature]])
        else:
            return Counter(self.tree).most_common(1)[0][0]

    def predict(self, data):
        return data.apply(lambda row: self.predict_row(row, self.tree), axis=1)


class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, sample_size=0.8):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.trees = []

    def bootstrap_sample(self, data):
        indices = np.random.choice(data.index, size=int(len(data) * self.sample_size), replace=True)
        return data.loc[indices]

    def fit(self, data, target, features):
        for _ in range(self.n_trees):
            sample = self.bootstrap_sample(data)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(sample, target, features)
            self.trees.append(tree)

    def predict(self, data):
        tree_preds = np.array([tree.predict(data) for tree in self.trees])
        majority_votes = [Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(data.shape[0])]
        return majority_votes

# Function for preprocessing data
def preprocess_data(df):
    mappings = {
        'Gender': {'Female': 0, 'Male': 1},
        'City': {'Chicago': 0, 'Houston': 1, 'Los Angeles': 2, 'Miami': 3, 'New York': 4, 'San Francisco': 5},
        'Membership Type': {'Bronze': 0, 'Gold': 1, 'Silver': 2},
        'Discount Applied': {False: 0, True: 1},
        'Satisfaction Level': {'Neutral': 0, 'Satisfied': 1, 'Unsatisfied': 2}
    }

    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    df = df.fillna(df.median())
    return df

# Load model
model_file = r'random_forest_model_90.pkl'
try:
    rf_model = joblib.load(model_file)
except Exception as e:
    st.error(f"Error saat memuat model: {e}")
    st.stop()

# Sidebar untuk mengunggah file
st.sidebar.title("Segment Pro")
uploaded_file = st.sidebar.file_uploader("Unggah File CSV", type="csv")

# Define consistent soft color mapping for each segment
color_mapping = {
    'Neutral': '#a8d5ba',      # Soft Green
    'Satisfied': '#ffe066',    # Soft Yellow
    'Unsatisfied': '#c9a9e3'   # Soft Lavender
}

# Extract colors as a list for charts that require a palette
color_palette = [color_mapping[key] for key in ['Neutral', 'Satisfied', 'Unsatisfied']]
# Fungsi untuk visualisasi tambahan
# Additional Visualizations
# Fungsi untuk visualisasi tambahan dan memberikan insight
def additional_visualizations(data):
    st.write("### Hubungan Antar Fitur dengan Hasil Klasifikasi dan Insight")
    features = ['Gender', 'Membership Type', 'Total Spend', 'Discount Applied', 'City']
    for feature in features:
        if feature in data.columns:
            st.write(f"#### Pengaruh {feature} terhadap Hasil Klasifikasi")
            
            # Visualisasi dengan countplot
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(
                data=data,
                x=feature,
                hue='Prediction',
                palette=color_palette,
                ax=ax
            )
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Menambahkan insight berdasarkan analisis data
            st.write("##### Insight:")
            if feature == 'Gender':
                st.write("""
                - Wanita (Female) lebih banyak berada dalam kategori **Satisfied** dibandingkan pria (Male).
                - Pria cenderung lebih sering berada dalam kategori **Neutral** atau **Unsatisfied**.
                """)
            elif feature == 'Membership Type':
                st.write("""
                - Pelanggan dengan **Gold Membership** lebih dominan dalam kategori **Satisfied**, menunjukkan layanan premium memberikan pengalaman yang lebih baik.
                - **Silver Membership** dominan di kategori **Neutral**, menandakan pengalaman cukup baik namun masih bisa ditingkatkan.
                - **Bronze Membership** banyak berada dalam kategori **Unsatisfied**, kemungkinan karena manfaat keanggotaan yang terbatas.
                """)
            elif feature == 'Total Spend':
                st.write("""
                - Pelanggan dengan **pengeluaran tinggi** dominan dalam kategori **Satisfied**.
                - Pelanggan dengan **pengeluaran rendah** sering berada dalam kategori **Neutral** atau **Unsatisfied**.
                - Hal ini menunjukkan hubungan langsung antara pengeluaran lebih besar dengan tingkat kepuasan yang tinggi.
                """)
            elif feature == 'Discount Applied':
                st.write("""
                - Pelanggan yang **mendapat diskon** (Discount Applied = True) lebih sering berada dalam kategori **Satisfied**.
                - Tanpa diskon, pelanggan lebih sering berada di kategori **Neutral** atau **Unsatisfied**.
                - Diskon memberikan rasa nilai tambah pada transaksi, sehingga meningkatkan kepuasan.
                """)
            elif feature == 'City':
                st.write("""
                - Pelanggan dari **San Francisco** dan **New York** lebih banyak berada dalam kategori **Satisfied**.
                - Pelanggan di **Houston** dan **Miami** memiliki distribusi merata antara kategori **Neutral** dan **Unsatisfied**.
                - Di **Chicago** dan **Los Angeles**, sebagian besar pelanggan berada dalam kategori **Neutral**, menandakan perlunya perbaikan layanan atau penawaran untuk meningkatkan kepuasan.
                """)

# Tambahkan bagian ini pada main loop Anda setelah `additional_visualizations` dipanggil.
def segment_analysis(data):
    st.write("### Analisis Segmentasi")
    st.write("Berikut adalah beberapa insight berdasarkan data segmentasi yang tersedia:")
    
    # Distribusi segmen
    segment_counts = data['Prediction'].value_counts()
    for segment, count in segment_counts.items():
        st.write(f"- **{segment}**: {count} pelanggan")

    # Insight tambahan (jika ada logika tambahan)
    st.write("""
    - Pelanggan dengan kepuasan **Satisfied** cenderung lebih loyal dan memiliki potensi lebih besar untuk upselling.
    - Pelanggan dengan kepuasan **Unsatisfied** memerlukan perhatian khusus untuk meningkatkan pengalaman mereka.
    """)


# Fungsi untuk menampilkan rekomendasi
def show_recommendations(segment):
    if segment == 'Neutral':
        st.write("""
        **Rekomendasi untuk Pelanggan dengan Kepuasan Netral:**
        - **Program Loyalitas**: Tawarkan insentif seperti diskon atau program poin untuk meningkatkan pengeluaran dan keterlibatan pelanggan.
        - **Penyesuaian Penawaran**: Personalisasikan penawaran berdasarkan data preferensi untuk menarik perhatian mereka.
        - **Feedback Langsung**: Kumpulkan umpan balik untuk memahami kebutuhan yang belum terpenuhi.
        """)
    elif segment == 'Satisfied':
        st.write("""
        **Rekomendasi untuk Pelanggan dengan Kepuasan Tinggi:**
        - **Retensi Pelanggan**: Pastikan pengalaman tetap konsisten dengan menawarkan layanan premium atau program keanggotaan eksklusif.
        - **Upselling dan Cross-selling**: Promosikan produk atau layanan tambahan untuk meningkatkan nilai transaksi.
        - **Ambassador Program**: Jadikan pelanggan dalam segmen ini sebagai duta merek dengan memberikan penghargaan atas rekomendasi mereka.
        """)
    elif segment == 'Unsatisfied':
        st.write("""
        **Rekomendasi untuk Pelanggan Tidak Puas:**
        - **Strategi Pemulihan**: Hubungi pelanggan secara proaktif dengan penawaran khusus untuk menarik mereka kembali.
        - **Analisis Masalah**: Identifikasi dan atasi penyebab ketidakpuasan berdasarkan data dan umpan balik.
        - **Komunikasi Personal**: Tunjukkan perhatian dengan mengirimkan pesan atau email yang menawarkan solusi untuk pengalaman mereka yang buruk.
        """)

# Fungsi untuk mengembalikan nilai asli (inverse mapping)
def inverse_preprocess_data(df):
    inverse_mappings = {
        'Gender': {0: 'Female', 1: 'Male'},
        'City': {0: 'Chicago', 1: 'Houston', 2: 'Los Angeles', 3: 'Miami', 4: 'New York', 5: 'San Francisco'},
        'Membership Type': {0: 'Bronze', 1: 'Gold', 2: 'Silver'},
        'Discount Applied': {0: False, 1: True},
        'Satisfaction Level': {0: 'Neutral', 1: 'Satisfied', 2: 'Unsatisfied'}
    }

    for col, inverse_mapping in inverse_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(inverse_mapping)

    return df

# Main Interface
st.title("Segmentasi Pelanggan Berdasarkan Tingkat Kepuasan")
st.write("""
Aplikasi ini menampilkan segmen pelanggan berdasarkan prediksi kepuasan dan rekomendasi untuk setiap segmen.
Silakan pilih segmen untuk melihat detail ID pelanggan dan rekomendasi.
""")

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)

        if data.empty:
            st.error("Dataset yang diunggah kosong.")
        else:
            # Preprocessing
            data_preprocessed = preprocess_data(data)

            # Melakukan prediksi
            try:
                features = ['Gender', 'Membership Type', 'Total Spend', 'Discount Applied', 'City']
                predictions = rf_model.predict(data_preprocessed[features])

                # Konversi hasil prediksi ke label yang sesuai
                label_mapping = {0: 'Neutral', 1: 'Satisfied', 2: 'Unsatisfied'}
                predictions_labels = [label_mapping.get(pred, 'Neutral') for pred in predictions]

                # Menambahkan kolom Customer ID dan Prediksi
                data_preprocessed['Prediction'] = predictions_labels
                data_preprocessed['Customer ID'] = data['Customer ID']

                # Mengembalikan nilai asli (kata-kata) untuk analisis
                data_analyzed = inverse_preprocess_data(data_preprocessed.copy())

                # Pilihan untuk mengunduh hasil
                csv = data_preprocessed[['Customer ID', 'Prediction']].to_csv(index=False)
                st.download_button(
                    label="Unduh Hasil Prediksi",
                    data=csv,
                    file_name="hasil_klasifikasi_segmentasi.csv",
                    mime="text/csv",
                )

                # Analisis Chart: Distribusi Segmentasi
                st.write("### Analisis Distribusi Segmentasi:")
                segment_counts = data_preprocessed['Prediction'].value_counts()

                # Plot Pie Chart
                fig, ax = plt.subplots()
                ax.pie(
                    segment_counts, 
                    labels=segment_counts.index, 
                    autopct='%1.1f%%', 
                    startangle=90, 
                    colors=[color_mapping[label] for label in segment_counts.index]
                )
                ax.axis('equal')
                st.pyplot(fig)

                # Membuat Bar Chart dengan warna dan menambahkan tulisan "count"
                fig, ax = plt.subplots()
                bars = ax.bar(segment_counts.index, segment_counts.values, color=[color_mapping[label] for label in segment_counts.index])
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom')

                ax.set_xlabel('Segmen Kepuasan')
                ax.set_ylabel('Jumlah Pelanggan')
                ax.set_title('Distribusi Pelanggan Berdasarkan Kepuasan')
                st.pyplot(fig)

                # Tampilkan ID pelanggan untuk semua segmen dalam tabel terpisah
                st.write("### ID Pelanggan Berdasarkan Segmen")
                segments = ['Neutral', 'Satisfied', 'Unsatisfied']
                columns = st.columns(len(segments))

                for col, segment in zip(columns, segments):
                    selected_ids = data_preprocessed[data_preprocessed['Prediction'] == segment]['Customer ID']
                    with col:
                        st.write(f"**{segment}**")
                        segment_table = pd.DataFrame({'ID': selected_ids.values})
                        st.dataframe(segment_table, height=200)

                # Mengembalikan nilai asli untuk analisis lebih lanjut
                data_analyzed = inverse_preprocess_data(data_preprocessed)

                # Visualisasi tambahan
                additional_visualizations(data_analyzed)

                # Gabungkan Analisis Kebutuhan Per Kelas dan Rekomendasi
                st.write("### Analisis dan Rekomendasi Berdasarkan Segmen")
                combined_choice = st.selectbox("Pilih Segmen untuk Analisis dan Rekomendasi", ['Neutral', 'Satisfied', 'Unsatisfied'])

                if combined_choice:
                    segment_data = data_analyzed[data_analyzed['Prediction'] == combined_choice]

                    # Analisis Kebutuhan Per Kelas
                    st.write(f"#### Analisis Kebutuhan untuk Segmen: {combined_choice}")
                    if not segment_data.empty:
                        avg_spending = segment_data['Total Spend'].mean()
                        avg_items = segment_data['Items Purchased'].mean()
                        avg_rating = segment_data['Average Rating'].mean()

                        st.write(f"- **Rata-rata pengeluaran**: ${avg_spending:.2f}")
                        st.write(f"- **Rata-rata jumlah barang dibeli**: {avg_items:.1f}")
                        st.write(f"- **Rata-rata rating**: {avg_rating:.1f}")
                    else:
                        st.warning(f"Tidak ada data untuk segmen {combined_choice}.")

                    # Rekomendasi Pelanggan
                    st.write(f"#### Rekomendasi untuk Segmen: {combined_choice}")
                    show_recommendations(combined_choice)

            except Exception as e:
                st.error(f"Error saat melakukan prediksi: {e}")
    except Exception as e:
        st.error(f"Error saat membaca file CSV: {e}")

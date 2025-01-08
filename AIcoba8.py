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


# Load Model
model_file = r'random_forest_model_90.pkl'
try:
    rf_model = joblib.load(model_file)
except Exception as e:
    st.error(f"Error saat memuat model: {e}")
    st.stop()

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

# Inverse mapping to original values
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

# Define consistent soft color mapping for segments
color_mapping = {
    'Neutral': '#a8d5ba',
    'Satisfied': '#ffe066',
    'Unsatisfied': '#c9a9e3'
}

# Main Interface
if "data_uploaded" not in st.session_state:
    st.session_state.data_uploaded = False

if not st.session_state.data_uploaded:
    st.title("Segmentasi Pelanggan Berdasarkan Tingkat Kepuasan")
    st.write("Unggah file CSV Anda untuk memulai analisis segmentasi.")
    
    uploaded_file = st.file_uploader("Unggah File CSV", type="csv")
    
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)

            if data.empty:
                st.error("Dataset yang diunggah kosong.")
            else:
                st.session_state.data = data
                st.session_state.data_uploaded = True
                st.experimental_rerun()
        except Exception as e:
            st.error(f"Error saat membaca file CSV: {e}")
else:
    st.title("Segmentasi Pelanggan Berdasarkan Tingkat Kepuasan")
    st.write("Data telah diunggah. Anda dapat kembali ke halaman awal untuk memasukkan data baru.")
    
    if st.button("Kembali ke Halaman Awal"):
        st.session_state.data_uploaded = False
        st.session_state.data = None
        st.experimental_rerun()

    data = st.session_state.data

    # Penjelasan Segmentasi
    st.write("""
    Segmentasi pelanggan membantu mengelompokkan pelanggan berdasarkan tingkat kepuasan mereka.
    Klasifikasi dilakukan menjadi tiga kategori:
    - **Neutral**: Pelanggan dengan kepuasan rata-rata.
    - **Satisfied**: Pelanggan dengan tingkat kepuasan tinggi.
    - **Unsatisfied**: Pelanggan dengan tingkat kepuasan rendah.
    """)

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

        # Mengembalikan nilai asli untuk analisis lebih lanjut
        data_analyzed = inverse_preprocess_data(data_preprocessed.copy())

        # Sidebar Pilihan Analisis
        st.sidebar.title("Pilih Analisis")
        analysis_choice = st.sidebar.radio(
            "Tampilkan hasil analisis:",
            (
                "Analisis Distribusi Segmentasi",
                "ID Pelanggan Berdasarkan Segmen",
                "Hubungan Antar Fitur dengan Hasil Klasifikasi dan Insight",
                "Analisis dan Rekomendasi Berdasarkan Segmen"
            )
        )

        if analysis_choice == "Analisis Distribusi Segmentasi":
            # Analisis Distribusi Segmentasi
            st.write("### Analisis Distribusi Segmentasi")
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

        elif analysis_choice == "ID Pelanggan Berdasarkan Segmen":
            # Tampilkan ID Pelanggan Berdasarkan Segmen
            st.write("### ID Pelanggan Berdasarkan Segmen")
            segments = ['Neutral', 'Satisfied', 'Unsatisfied']
            columns = st.columns(len(segments))

            for col, segment in zip(columns, segments):
                selected_ids = data_preprocessed[data_preprocessed['Prediction'] == segment]['Customer ID']
                with col:
                    st.write(f"**{segment}**")
                    segment_table = pd.DataFrame({'ID': selected_ids.values})
                    st.dataframe(segment_table, height=200)

        elif analysis_choice == "Hubungan Antar Fitur dengan Hasil Klasifikasi dan Insight":
            # Hubungan Antar Fitur
            st.write("### Hubungan Antar Fitur dengan Hasil Klasifikasi dan Insight")
            features = ['Gender', 'Membership Type', 'Total Spend', 'Discount Applied', 'City']
            for feature in features:
                if feature in data_analyzed.columns:
                    st.write(f"#### Pengaruh {feature} terhadap Hasil Klasifikasi")

                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.countplot(
                        data=data_analyzed,
                        x=feature,
                        hue='Prediction',
                        palette=color_mapping,
                        ax=ax
                    )
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

        elif analysis_choice == "Analisis dan Rekomendasi Berdasarkan Segmen":
            # Analisis dan Rekomendasi Berdasarkan Segmen
            st.write("### Analisis dan Rekomendasi Berdasarkan Segmen")
            combined_choice = st.selectbox("Pilih Segmen", ['Neutral', 'Satisfied', 'Unsatisfied'])

            segment_data = data_analyzed[data_analyzed['Prediction'] == combined_choice]

            st.write(f"#### Analisis untuk Segmen: {combined_choice}")
            if not segment_data.empty:
                avg_spending = segment_data['Total Spend'].mean()
                st.write(f"- **Rata-rata pengeluaran**: ${avg_spending:.2f}")
            else:
                st.warning(f"Tidak ada data untuk segmen {combined_choice}.")

    except Exception as e:
        st.error(f"Error saat melakukan prediksi: {e}")

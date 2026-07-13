# Skripsi: Analisis Perbandingan Baseline RAG dan RAG-Fusion pada Dokumen Rekam Medis

Proyek ini merupakan implementasi penelitian untuk mengevaluasi dan membandingkan dua arsitektur Retrieval-Augmented Generation (RAG): **Baseline RAG** dan **RAG-Fusion**, khususnya dalam konteks analisis dokumen rekam medis dan nutrisi di Indonesia.

## 1. Latar Belakang

Dokumen medis memiliki karakteristik bahasa yang teknis, penuh singkatan, dan memerlukan akurasi informasi yang sangat tinggi. Tantangan utama penggunaan Large Language Model (LLM) pada domain ini adalah risiko halusinasi. Arsitektur RAG digunakan untuk memitigasi hal ini dengan menyediakan konteks nyata dari rekam medis. Namun, pencarian vektor tunggal (baseline) seringkali gagal menangkap nuansa pertanyaan medis yang kompleks, sehingga teknik **RAG-Fusion** diusulkan sebagai solusi untuk meningkatkan relevansi dokumen yang diambil.

## 2. Tujuan Penelitian

1.  Mengimplementasikan pipeline RAG Baseline dan RAG-Fusion yang dioptimalkan untuk dokumen medis berbahasa Indonesia.
2.  Menganalisis peningkatan kualitas retrieval melalui teknik _Multi-Query Generation_ dan _Reciprocal Rank Fusion_ (RRF).
3.  Mengevaluasi performa kedua sistem menggunakan framework evaluasi modern berbasis LLM.

## 3. Sumber Data dan Preprocessing

### A. Sumber Data (Raw Data)

Data mentah yang digunakan dalam penelitian ini adalah laporan medis pasien pada program dietetic intership mahasiswa Fakultas Ilmu Kesehatan dan Kedokteran Universitas Brawijaya dalam format **PDF** dengan jumlah file sebanyak 106. Laporan ini mencakup data klinis, diagnosis medis, serta catatan intervensi gizi.

### B. Tahapan Preprocessing

Proses pengolahan rekam medis dari format cetak (PDF) menjadi representasi teks bersih yang siap diindeks menggunakan tahapan sistematis berikut:

1.  **Parsing Layout-Aware (PDF to MD)**: Mengonversi berkas dokumen rekam medis mentah (format PDF) ke dalam format Markdown menggunakan library parser `Docling`. Penggunaan `Docling` krusial untuk menjaga hirarki dokumen, melacak bagian sub-bab (seperti identitas pasien, diagnosis, dan NCP), serta mengekstrak data tabel klinis kompleks agar tetap terstruktur rapi tanpa merusak relasi baris dan kolom.
2.  **Text Cleaning & Normalization**: Menggunakan skrip pembersihan teks untuk menghapus tag noise yang dihasilkan oleh parser PDF (seperti penanda gambar `<!-- image -->` dan penanda formula gagal decode `<!-- formula-not-decoded -->`). Seluruh teks juga dibersihkan dari spasi berlebih, tabulasi berulang, dan baris kosong yang tidak perlu menggunakan proses pemangkasan teks baris demi baris demi meminimalisasi konsumsi token LLM.
3.  **Field Extraction & Metadata Enrichment**: Mengekstrak data rekam medis utama menggunakan LLM (`gpt-4o-mini-2024-07-18`) berbasis _Structured Output_. Teks pasien dipotong menggunakan fungsi regex `get_section` untuk mengambil bab NCP (Nutritional Care Process) dengan batas maksimal 6.000 karakter, lalu diarahkan ke schema objek Pydantic (`DataMedis` dan `Antropometri`) guna mengambil diagnosis utama pasien serta data fisik antropometri (seperti gender, umur, berat badan, tinggi badan, IMT, dan LILA). Metadata klinis ini disimpan dalam format `.pkl` dan `.json` untuk memperkaya data chunk saat indexing.

### C. Pembuatan Data Sintetis (Golden Dataset)

Untuk mengevaluasi performa arsitektur RAG secara obyektif tanpa bergantung pada pelabelan manual oleh pakar gizi yang memakan waktu, dataset evaluasi (_Golden Dataset_) dibuat secara sintetis melalui alur logika berikut:

1.  **Grouping Document by Patient**: Seluruh dokumen chunks dari data pickle dikelompokkan berdasarkan metadata sumber aslinya (`source` rekam medis pasien). Langkah ini penting guna menjamin bahwa LLM hanya akan mengaitkan informasi klinis dari pasien yang sama, bukan mencampur informasi antar pasien yang berbeda.
2.  **Multi-Hop Context Selection**: Dari dokumen pasien yang sama (yang memiliki minimal 2 chunks), dipilih 2 hingga 3 chunks secara acak sebagai gabungan konteks sumber pertanyaan. Metode _multi-hop_ ini memaksa LLM untuk menyusun pertanyaan yang membutuhkan korelasi informasi lintas bab dalam rekam medis tersebut.
3.  **LLM-based QA Generation & Constraint Rules**: Konteks gabungan dikirim ke API OpenAI `gpt-4o-mini` dengan mengaktifkan mode keluaran JSON (`response_format={"type": "json_object"}`). Pembuatan QA dikontrol ketat menggunakan aturan instruksi (_Prompt Engineering_):
    - **Aturan Positif**: Wajib menghasilkan pertanyaan bernuansa penalaran klinis yang menghubungkan minimal dua bagian informasi (misal: pengaruh diagnosis penyakit terhadap rencana intervensi atau edukasi gizi yang dirumuskan).
    - **Aturan Negatif**: Dilarang membuat pertanyaan pencarian langsung (_simple lookup_) seperti menanyakan diagnosis medis secara langsung, angka laboratorium secara mentah, atau berat/tinggi badan pasien untuk menghindari bias evaluasi yang terlalu mudah.
4.  **Serialization of Golden Dataset**: Pasangan tanya-jawab yang lolos validasi disimpan dalam file `nutrition_qa_golden_dataset.json` dengan struktur kaya informasi, mencakup: `question` (pertanyaan sintetis), `ground_truth` (jawaban referensi dari LLM), `gold_contexts` (teks asli chunks yang digunakan sebagai basis pertanyaan), dan `chunk_metadata` (metadata chunk penunjuk indeks dokumen).

### D. Tahapan Indexing

Setelah dokumen rekam medis melewati tahap preprocessing dan pembersihan teks, dokumen siap untuk dimasukkan ke dalam basis data vektor (Qdrant). Proses indexing dilakukan melalui tahapan berikut:

1.  **Structured Metadata Extraction & Enrichment**: Menggunakan LLM (`gpt-4o-mini`) dengan fitur _Structured Output_ (berdasarkan schema `DataMedis` dan `Antropometri`) untuk mengekstraksi data klinis secara otomatis. Data yang diekstraksi mencakup **diagnosis medis** dan **data antropometri** (jenis kelamin, usia, berat badan, tinggi badan, lingkar kepala, lingkar lengan atas, serta indeks massa tubuh).
2.  **Document Chunking (Markdown Header Splitting)**: Memecah dokumen Markdown menggunakan `MarkdownHeaderTextSplitter` berdasarkan hierarki header (`#`, `##`, `###`). Pendekatan ini dipilih agar dokumen terpecah secara logis sesuai struktur laporan medis tanpa merusak kesatuan informasi di setiap bagian.
3.  **Injeksi Informasi Klinis (Content Enrichment)**: Menyisipkan informasi klinis hasil ekstraksi (Diagnosis Medis & Assessment Fisik/Antropometri) ke bagian paling atas dari isi teks setiap chunk (di bawah format penulisan `INFO KLINIS PASIEN:` dan `ISI LAPORAN:`). Hal ini menjamin bahwa setiap representasi potongan teks membawa informasi konteks pasien yang lengkap ketika dicari.
4.  **Serialization & Backup**: Menyimpan seluruh chunks hasil rekonstruksi dan pengayaan metadata ke dalam file backup berformat pickle (`.pkl`) untuk memudahkan pemuatan ulang tanpa memanggil API ekstraksi LLM kembali.
5.  **Embedding Generation**: Memasukkan potongan dokumen ke dalam model embedding dengan optimasi kuantisasi 4-bit (`BitsAndBytesConfig`) untuk menghemat konsumsi memori GPU (VRAM) dan mempercepat inferensi pada CUDA.
6.  **Vector Store Upload (Qdrant)**: Mengunggah vektor embedding ke basis data vektor Qdrant secara bertahap (_batching_) dengan ukuran batch 4 untuk stabilitas koneksi. Menggunakan metrik jarak **Cosine Distance** untuk penentuan kemiripan semantik, serta menerapkan manajemen memori otomatis dengan menghapus _cache_ GPU (`torch.cuda.empty_cache()` dan `gc.collect()`) secara periodik setelah setiap batch berhasil diunggah.

## 4. Arsitektur Sistem

Penelitian ini membandingkan dua arsitektur sistem Retrieval-Augmented Generation (RAG) untuk memproses kueri rekam medis:

### A. Baseline RAG

1.  **Context-Aware Querying**: Menerima kueri atau pertanyaan dari pengguna, lalu menambahkan instruksi embedding khusus di awal kueri, yaitu: `"Representasikan pertanyaan medis ini untuk pencarian rekam medis yang relevan: "` sebelum dikonversi menjadi vektor kueri.
2.  **Dense Retrieval**: Melakukan pencarian vektor tunggal pada database Qdrant dengan menghitung skor kemiripan kosinus (_Cosine Similarity_) antara kueri dan chunk dokumen. Pencarian ini mengembalikan sejumlah _Top-K_ dokumen (default $K=3$). Sistem juga mendukung penyaringan metadata secara langsung (`metadata.medical_diagnosis`) untuk membatasi ruang pencarian.
3.  **Generation**:
    - **Prompt System**: Konteks dokumen yang diperoleh diformat dengan pemisah yang jelas beserta sumber berkasnya (contoh: `--- DOKUMEN 1 (Sumber: pasien_A.md) ---`).
    - **Local LLM Inference**: Menggunakan model `Qwen/Qwen3-4B-Instruct-2507` yang dimuat pada GPU lokal dengan kuantisasi 4-bit (`BitsAndBytesConfig`, tipe data `nf4`, perhitungan `torch.float16`).
    - **Prompt Constraints**: Model diinstruksikan untuk bertindak sebagai asisten medis profesional, dilarang keras mencampuradukkan data klinis antar dokumen pasien yang berbeda, mengutamakan detail data klinis asli, dan wajib menjawab "Informasi tidak lengkap." jika konteks tidak mencukupi untuk meminimalkan halusinasi medis. Model dipotong tepat pada tag ChatML `<|im_end|>` dan `<|endoftext|>`.

### B. RAG-Fusion

1.  **Query Expansion (Multi-Query Generation)**: Menggunakan LLM (`gpt-4o-mini`) dengan suhu rendah (`temperature=0.2`) untuk membongkar kueri pengguna menjadi 3 variasi kueri tambahan. Prompt dirancang untuk memecah pertanyaan kompleks, memunculkan sinonim istilah medis, atau mengekstrak akronim klinis (contoh: menerjemahkan singkatan gizi "BBLR" menjadi "Berat Badan Lahir Rendah"), dengan syarat wajib mempertahankan identitas klinis pasien (seperti diagnosis, usia, dan gender).
2.  **Parallel Multi-Retrieval**: Menjalankan proses pencarian vektor dense secara paralel pada database Qdrant untuk seluruh kueri hasil ekspansi (total 4 kueri: 1 kueri asli + 3 kueri ekspansi). Setiap pencarian mengembalikan kandidat dokumen sebanyak $2 \times K$ (default 6 dokumen per kueri).
3.  **Reciprocal Rank Fusion (RRF)**: Menggabungkan seluruh hasil pencarian dari multi-query tersebut. Untuk menghindari bias duplikasi berkas, dokumen dikonversi menjadi string JSON (`model_dump_json()`) untuk dijadikan kunci unik. Skor RRF dihitung dengan rumus matematika:
    $$Score(d \in D) = \sum_{q \in Q} \frac{1}{rank(d, q) + k}$$
    di mana $rank(d, q)$ adalah peringkat dokumen $d$ pada hasil pencarian kueri $q$, dan $k$ adalah konstanta penyeimbang peringkat (ditetapkan sebesar $60$). Dokumen diurutkan berdasarkan skor RRF tertinggi dan dipotong sebanyak $K$ dokumen teratas.
4.  **Generation**: Dokumen hasil reranking RRF diformat ke dalam template konteks medis dan disuapkan ke LLM lokal `Qwen` (konfigurasi 4-bit) untuk menghasilkan jawaban komprehensif yang memadukan informasi dari berbagai perspektif kueri ekspansi.

## 5. Komponen Teknologi

- **Generator LLM**: `Qwen/Qwen3-4B-Instruct-2507` (Dijalankan secara lokal pada CUDA dengan kuantisasi 4-bit menggunakan `BitsAndBytesConfig`).
- **Embedding Model**: `intfloat/multilingual-e5-large` (atau model embedding yang dikonfigurasi melalui HuggingFace).
- **Vector Database**: Qdrant (menggunakan metrik jarak Cosine Distance).
- **Orchestration**: LangChain.
- **Framework Evaluasi**: DeepEval.

## 6. Metodologi Evaluasi

Proses evaluasi komparatif dijalankan menggunakan notebook [evaluation_deepeval.ipynb](file:///D:/Kuliah/SKRIPSI/code/playground/notebooks/evaluation_deepeval.ipynb) untuk menguji performa Baseline RAG vs. RAG-Fusion pada 50 sampel data kueri evaluasi. Evaluasi ini mencakup dua kategori utama:

### A. Metrik Retrieval (Tradisional & Programmatic)

Dihitung secara terprogram dengan mencocokkan ID chunk dokumen yang diambil oleh sistem (`baseline_retrieved_ids` / `fusion_retrieved_ids`) terhadap ID kunci jawaban asli (`ground_truth_ids`):

- **MRR@K (Mean Reciprocal Rank @ K)**: Mengukur kualitas peringkat dokumen relevan pertama yang berhasil ditemukan oleh sistem. Nilai MRR dihitung dari kebalikan posisi peringkat (_reciprocal rank_) hit pertama:
  $$MRR@K = \frac{1}{\min_{i} (\text{posisi hit pertama pada indeks } i)}$$
- **Contextual Precision**: Menilai efektivitas mesin pencari dalam memosisikan dokumen relevan pada urutan peringkat teratas menggunakan `ContextualPrecisionMetric(threshold=0.5)`.
- **Contextual Recall**: Menilai kelengkapan informasi dalam konteks pencarian menggunakan `ContextualRecallMetric(threshold=0.5)` dengan membandingkan poin informasi kunci pada `expected_output` terhadap teks di `retrieval_context`.

### B. Metrik Generation (LLM-as-a-Judge - DeepEval)

Menggunakan framework **DeepEval** untuk melakukan evaluasi semantik berbasis LLM dengan model `gpt-4o-mini` sebagai _judge_. Setiap sampel dikonversi menjadi objek `LLMTestCase` yang memuat parameter `input` (pertanyaan), `actual_output` (jawaban bersih RAG), `expected_output` (jawaban acuan _ground truth_), dan `retrieval_context` (daftar teks dokumen pendukung). Pengujian berjalan secara paralel menggunakan `AsyncConfig(max_concurrent=3, throttle_value=2)` dengan metrik berikut:

- **Faithfulness**: Mengukur tingkat kepatuhan jawaban terhadap konteks yang diberikan (bebas halusinasi medis). Menggunakan `FaithfulnessMetric(threshold=0.5)` untuk mendeteksi apakah setiap klaim fakta dalam jawaban sistem didukung penuh oleh teks dalam `retrieval_context`.
- **Answer Relevancy**: Mengukur seberapa tepat jawaban LLM dalam merespons kueri masukan menggunakan `AnswerRelevancyMetric(threshold=0.5)`. Skor akan dipotong jika jawaban mengandung pernyataan bertele-tele atau di luar topik.

## 7. Cara Menjalankan

1.  **Instalasi**: `pip install -r requirements.txt`
2.  **Preprocessing**: Jalankan `pdf_to_md_parser.py` diikuti `text_cleaning.py`.
3.  **Indexing**: Jalankan `indexing_medical_v2.py` untuk mengekstraksi informasi klinis, melakukan chunking, dan mengunggah dokumen ke vector database Qdrant.
4.  **Synthetic Data Generation**: Jalankan `synthetic_data_generator.py` untuk menghasilkan dataset evaluasi (_golden dataset_).
5.  **Evaluasi**: Jalankan notebooks/evaluation_deepeval.ipynb untuk mendapatkan skor performa perbandingan.

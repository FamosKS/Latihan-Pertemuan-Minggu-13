import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier

from skimage.feature import hog
from skimage.measure import moments_central, moments_normalized, moments_hu

warnings.filterwarnings('ignore')

# ==========================================
# 1. PERSIAPAN DATASET & EKSTRAKSI FITUR
# ==========================================
def load_and_extract_features():
    print("1. Memuat Dataset dan Ekstraksi Fitur...")
    # Load dataset MNIST (8x8)
    digits = load_digits()
    
    # Ambil hanya 1000 sampel pertama sesuai permintaan
    X_raw = digits.images[:1000]
    y = digits.target[:1000]
    
    X_features = []
    
    for img in X_raw:
        # Fitur 1: HOG (Histogram of Oriented Gradients)
        # Karena citra 8x8 sangat kecil, kita gunakan cell 4x4
        fd_hog = hog(img, orientations=8, pixels_per_cell=(4, 4),
                     cells_per_block=(1, 1), visualize=False)
        
        # Fitur 2: Hu Moments (Shape Feature)
        # Menangkap momen bentuk invarian terhadap translasi/skala/rotasi
        m = moments_central(img)
        nu = moments_normalized(m)
        hu = moments_hu(nu)
        
        # Gabungkan fitur HOG (32 dimensi) dan Hu Moments (7 dimensi)
        combined_features = np.hstack([fd_hog, hu])
        X_features.append(combined_features)
        
    X = np.array(X_features)
    print(f"Bentuk Dataset Asli: {X_raw.shape}")
    print(f"Bentuk Fitur Ekstraksi (HOG + Hu Moments): {X.shape}\n")
    return X, y

# ==========================================
# 2. PELATIHAN & HYPERPARAMETER TUNING
# ==========================================
def train_and_tune(X_train, y_train):
    print("2. Hyperparameter Tuning dengan GridSearchCV (5-Fold Stratified CV)...")
    
    # a. KNN Tuning
    knn_params = {
        'n_neighbors': [1, 3, 5, 7, 9, 11],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, scoring='accuracy', n_jobs=-1)
    
    start_time = time.time()
    knn_grid.fit(X_train, y_train)
    knn_train_time = time.time() - start_time
    
    print(f"KNN Optimal: {knn_grid.best_params_} (Akurasi CV: {knn_grid.best_score_:.4f})")
    
    # b. SVM Tuning
    svm_params = [
        {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
        {'kernel': ['poly'], 'C': [0.1, 1, 10], 'degree': [3]},
        {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1]}
    ]
    svm_grid = GridSearchCV(SVC(probability=True, random_state=42), svm_params, cv=5, scoring='accuracy', n_jobs=-1)
    
    start_time = time.time()
    svm_grid.fit(X_train, y_train)
    svm_train_time = time.time() - start_time
    
    print(f"SVM Optimal: {svm_grid.best_params_} (Akurasi CV: {svm_grid.best_score_:.4f})\n")
    
    return knn_grid.best_estimator_, svm_grid.best_estimator_, knn_train_time, svm_train_time

# ==========================================
# 3. EVALUASI DAN VISUALISASI KINERJA
# ==========================================
def evaluate_models(models, X_test, y_test):
    print("3. Evaluasi Model pada Data Uji...")
    results = {}
    
    for name, model in models.items():
        # Inference Time
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        
        accuracy = model.score(X_test, y_test)
        
        print(f"\n--- Laporan Klasifikasi {name} ---")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Aktual')
        plt.xlabel('Prediksi')
        plt.show()
        
        results[name] = {'accuracy': accuracy, 'inf_time': inference_time, 'model': model}
        
    return results

def plot_roc_curve(svm_model, X_train, y_train, X_test, y_test):
    # Menggunakan SVM Optimal untuk visualisasi ROC Multi-class
    y_test_bin = label_binarize(y_test, classes=range(10))
    ovr = OneVsRestClassifier(svm_model)
    ovr.fit(X_train, y_train)
    y_score = ovr.predict_proba(X_test)
    
    plt.figure(figsize=(8, 6))
    for i in range(10):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        plt.plot(fpr, tpr, lw=1.5, label=f'Class {i} (AUC = {auc(fpr, tpr):.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Multi-class (SVM Best Estimator)')
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.show()

# ==========================================
# 4. DECISION BOUNDARY & LEARNING CURVE
# ==========================================
def plot_pca_decision_boundary(knn_best, svm_best, X_scaled, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Buat instance baru untuk plotting di dimensi 2D
    svm_2d = SVC(**svm_best.get_params())
    knn_2d = KNeighborsClassifier(**knn_best.get_params())
    
    svm_2d.fit(X_pca, y)
    knn_2d.fit(X_pca, y)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
    
    for ax, model, name in zip(axes, [knn_2d, svm_2d], ['KNN Optimal', 'SVM Optimal']):
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='tab10')
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', edgecolor='k', s=20)
        ax.set_title(f'Decision Boundary (PCA 2D) - {name}')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
    
    plt.tight_layout()
    plt.show()

def plot_learning_curve(model, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    plt.title(f'Learning Curve - {title}')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.legend(loc="best")
    plt.grid(alpha=0.3)

# ==========================================
# PROGRAM UTAMA
# ==========================================
def main():
    print("=" * 50)
    print("KOMPARASI KNN vs SVM UNTUK PENGENALAN DIGIT")
    print("=" * 50)
    
    # 1. Ekstraksi Fitur
    X, y = load_and_extract_features()
    
    # Splitting & Scaling (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_scaled = scaler.transform(X) # Untuk visualisasi keseluruhan
    
    # 2. Tuning dan Training
    knn_best, svm_best, knn_t_time, svm_t_time = train_and_tune(X_train_scaled, y_train)
    
    # 3. Evaluasi Model
    models = {'KNN Optimal': knn_best, 'SVM Optimal': svm_best}
    eval_results = evaluate_models(models, X_test_scaled, y_test)
    
    # 4. ROC Curve
    plot_roc_curve(svm_best, X_train_scaled, y_train, X_test_scaled, y_test)
    
    # 5. Visualisasi Decision Boundary
    plot_pca_decision_boundary(knn_best, svm_best, X_scaled, y)
    
    # 6. Learning Curve
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_learning_curve(knn_best, X_scaled, y, "KNN Optimal")
    plt.subplot(1, 2, 2)
    plot_learning_curve(svm_best, X_scaled, y, "SVM Optimal")
    plt.tight_layout()
    plt.show()
    
    # 7. Rangkuman Komparasi
    print("\n" + "=" * 50)
    print("KESIMPULAN DAN ANALISIS KOMPARATIF")
    print("=" * 50)
    print(f"{'Metrik':<20} | {'KNN Optimal':<15} | {'SVM Optimal':<15}")
    print("-" * 55)
    print(f"{'Akurasi Uji':<20} | {eval_results['KNN Optimal']['accuracy']*100:.2f}%{'':<9} | {eval_results['SVM Optimal']['accuracy']*100:.2f}%")
    print(f"{'Waktu Training':<20} | {knn_t_time:.4f} detik{'':<3} | {svm_t_time:.4f} detik")
    print(f"{'Waktu Inference':<20} | {eval_results['KNN Optimal']['inf_time']:.4f} detik{'':<3} | {eval_results['SVM Optimal']['inf_time']:.4f} detik")
    
if __name__ == "__main__":
    main()
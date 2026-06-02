import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.filterwarnings('ignore')

# Latihan 2: SVM dengan berbagai kernel
def praktikum_svm_fruits():
    print("\nLATIHAN 2: SVM DENGAN BERBAGAI KERNEL")
    print("=" * 50)
    
    # Create synthetic fruit dataset
    def create_fruit_dataset(n_samples=300):
        np.random.seed(42)
        n_features = 20
        
        # Three fruit classes with different characteristics
        # Class 0: Apples (spherical, red)
        apples = np.random.randn(n_samples, n_features)
        apples[:, 0] += 2  # Red color feature
        apples[:, 1] += 1  # Roundness feature
        apples_labels = np.zeros(n_samples)
        
        # Class 1: Bananas (elongated, yellow)
        bananas = np.random.randn(n_samples, n_features)
        bananas[:, 0] += 1  # Yellow color feature
        bananas[:, 1] += 3  # Elongation feature
        bananas_labels = np.ones(n_samples)
        
        # Class 2: Oranges (spherical, orange)
        oranges = np.random.randn(n_samples, n_features)
        oranges[:, 0] += 1.5  # Orange color feature
        oranges[:, 1] += 1   # Roundness feature
        oranges[:, 2] += 2   # Texture feature
        oranges_labels = np.ones(n_samples) * 2
        
        X = np.vstack([apples, bananas, oranges])
        y = np.hstack([apples_labels, bananas_labels, oranges_labels])
        
        return X, y
    
    X, y = create_fruit_dataset(100)
    fruit_names = ['Apple', 'Banana', 'Orange']
    
    print(f"Dataset Shape: {X.shape}")
    print(f"Class Distribution: {np.bincount(y.astype(int))}")
    
    # Split and preprocess data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_scaled)
    
    # Visualize data in PCA space
    plt.figure(figsize=(10, 6))
    colors = ['red', 'yellow', 'orange']
    for i, color in enumerate(colors):
        plt.scatter(X_pca[y_train == i, 0], X_pca[y_train == i, 1], 
                    c=color, label=fruit_names[i], alpha=0.6, edgecolors='black')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Fruit Dataset in PCA Space')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Experiment with different SVM kernels
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    results = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, kernel in enumerate(kernels):
        if kernel == 'poly':
            svm_model = svm.SVC(kernel=kernel, degree=3, C=1.0, random_state=42)
        else:
            svm_model = svm.SVC(kernel=kernel, C=1.0, random_state=42)
        
        # Cross-validation
        cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5)
        
        # Train and test
        svm_model.fit(X_train_scaled, y_train)
        y_pred = svm_model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = svm_model.score(X_test_scaled, y_test)
        
        # Create meshgrid for decision boundary visualization
        if X_pca.shape[1] == 2:
            X0 = X_pca[:, 0]
            X1 = X_pca[:, 1]
            
            x_min, x_max = X0.min() - 1, X0.max() + 1
            y_min, y_max = X1.min() - 1, X1.max() + 1
            
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                np.arange(y_min, y_max, 0.1))
            
            # --- BAGIAN YANG DIPERBAIKI ---
            # Transform meshgrid points back to 20D scaled space
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            mesh_points_20d_scaled = pca.inverse_transform(mesh_points)
            
            # Predict directly (data is already in scaled space)
            Z = svm_model.predict(mesh_points_20d_scaled)
            Z = Z.reshape(xx.shape)
            # ------------------------------
            
            axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
            
            for i, color in enumerate(colors):
                axes[idx].scatter(X_pca[y_train == i, 0], X_pca[y_train == i, 1], 
                                c=color, label=fruit_names[i], alpha=0.7, edgecolors='black')
        
        axes[idx].set_title(f'SVM with {kernel.upper()} Kernel\nCV Acc: {cv_scores.mean():.3f}, Test Acc: {accuracy:.3f}')
        axes[idx].set_xlabel('PC1')
        axes[idx].set_ylabel('PC2')
        axes[idx].legend(loc='upper right')
        
        results[kernel] = {
            'cv_accuracy': cv_scores.mean(),
            'test_accuracy': accuracy,
            'model': svm_model
        }
    
    plt.tight_layout()
    plt.show()
    
    # Compare results
    print("\nSVM KERNEL COMPARISON:")
    print("-" * 40)
    print(f"{'Kernel':<10} {'CV Accuracy':<12} {'Test Accuracy':<12}")
    print("-" * 40)
    for kernel, result in results.items():
        print(f"{kernel:<10} {result['cv_accuracy']:<12.4f} {result['test_accuracy']:<12.4f}")
    
    # ROC Curve for best kernel
    best_kernel = max(results, key=lambda k: results[k]['test_accuracy'])
    print(f"\nBest Kernel: {best_kernel}")
    
    # Binarize labels for ROC curve
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    
    # Create OneVsRest classifier for ROC
    ovr_classifier = OneVsRestClassifier(
        svm.SVC(kernel=best_kernel, C=1.0, probability=True, random_state=42)
    )
    ovr_classifier.fit(X_train_scaled, y_train)
    y_score = ovr_classifier.predict_proba(X_test_scaled)
    
    # Compute ROC curve and ROC area for each class
    plt.figure(figsize=(10, 8))
    for i in range(len(fruit_names)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, 
                label=f'{fruit_names[i]} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - SVM with {best_kernel.upper()} Kernel')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Feature importance for linear kernel
    if best_kernel == 'linear':
        coef = results['linear']['model'].coef_
        plt.figure(figsize=(12, 6))
        
        for i in range(coef.shape[0]):
            plt.bar(range(coef.shape[1]), coef[i], alpha=0.6, 
                    label=f'Class {fruit_names[i]} vs Rest')
        
        plt.xlabel('Feature Index')
        plt.ylabel('Coefficient Value')
        plt.title('SVM Feature Importance (Linear Kernel)')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.show()
    
    return results, fruit_names

# Jalankan latihan 2
svm_results, fruit_names = praktikum_svm_fruits()
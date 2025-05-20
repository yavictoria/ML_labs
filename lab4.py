# Пункт 1. (опціонально) Створення віртуального середовища

# Пункт 2. Завантаження набору даних Fashion MNIST, підготовка і розділення

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2.a) Завантаження даних
train_df = pd.read_csv('fashion-mnist_train.csv')
test_df = pd.read_csv('fashion-mnist_test.csv')

# 2.b) Розділення на ознаки і мітки
X_train = train_df.drop('label', axis=1).values
y_train = train_df['label'].values
X_test = test_df.drop('label', axis=1).values
y_test = test_df['label'].values

# 2.c) Масштабування в діапазон [0, 1)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# 2.d) Назви класів для зручності виводу
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Виведемо приклад зображення
plt.imshow(X_train[0].reshape(28,28), cmap='gray')
plt.title(f"Клас: {y_train[0]} ({class_names[y_train[0]]})")
plt.axis('off')
plt.show()

# Пункт 3. Навчання класифікаторів

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# 3.a-f) Створюємо класифікатори
classifiers = [
    ("Логістична регресія", LogisticRegression(max_iter=500, solver='lbfgs', multi_class='auto')),
    ("Дерево рішень", DecisionTreeClassifier()),
    ("k-найближчих сусідів", KNeighborsClassifier(n_neighbors=5)),
    ("Метод опорних векторів", SVC()),
    ("Випадковий ліс", RandomForestClassifier(n_estimators=100)),
    ("Наївний Байєс", GaussianNB()),
    ("Лінійний дискримінантний аналіз", LinearDiscriminantAnalysis())
]

results = []

# Пункт 4. Оцінка кожного класифікатора
for name, clf in classifiers:
    print("="*70)
    print(f"4. Оцінка класифікатора: {name}")

    # 4.a) Вимірювання часу навчання
    start_train = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_train

    # 4.a) Вимірювання часу класифікації
    start_pred = time.time()
    y_pred = clf.predict(X_test)
    pred_time = time.time() - start_pred

    # 4.b) Оцінка точності
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точність: {accuracy:.4f}")

    # 4.c) Звіт по класифікації
    print("\nЗвіт по класифікації:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # 4.d) Матриця невідповідностей
    print("Матриця невідповідностей:")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation='vertical')
    plt.title(f"Confusion Matrix: {name}")
    plt.show()

    # 4.e) Класифікація 20 випадкових зразків з тесту
    print("20 випадкових зразків з тестового набору:")
    idxs = np.random.choice(len(X_test), size=20, replace=False)
    for i, idx in enumerate(idxs):
        true_label = y_test[idx]
        pred_label = y_pred[idx]
        print(f"{i+1:2d}. Очікувано: {class_names[true_label]:12s} | Передбачено: {class_names[pred_label]:12s} {'✅' if true_label==pred_label else '❌'}")
    print()

    # Для підсумкової таблиці
    results.append([name, f"{accuracy:.4f}", f"{train_time:.2f}", f"{pred_time:.2f}"])

    # Пункт 5. Таблиця результатів
import pandas as pd

# Налаштування pandas для повного виводу таблиці
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.colheader_justify', 'left')

results_df = pd.DataFrame(results, columns=[
    "Метод", "Точність класифікації", "Час навчання, с", "Час класифікації, с"
])

print("=" * 70)
print("Порівняльна таблиця результатів:")
print(results_df)

# Пункт 6. k-fold cross-validation для найкращого класифікатора
from sklearn.model_selection import StratifiedKFold

# 6.a) Знаходимо найкращий класифікатор за точністю
best_idx = np.argmax(results_df["Точність класифікації"].astype(float))
best_name, best_clf = classifiers[best_idx]
print(f"\nНайкращий класифікатор: {best_name}")

# 6.b) Визначаємо k так, щоб розмір fold ≈ розміру тестового набору
k = len(X_train) // len(X_test)
print(f"Використовується StratifiedKFold з k = {k}")

skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

fold = 1
fold_accuracies = []

for train_index, val_index in skf.split(X_train, y_train):
    X_tr, X_val = X_train[train_index], X_train[val_index]
    y_tr, y_val = y_train[train_index], y_train[val_index]
    best_clf.fit(X_tr, y_tr)
    y_val_pred = best_clf.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    fold_accuracies.append(acc)
    print(f"\nFold {fold}: Точність = {acc:.4f}")
    try:
        print(classification_report(y_val, y_val_pred, target_names=class_names))
    except Exception as e:
        print(f"[Помилка в classification_report]: {e}")

    fold += 1

print(f"\nСередня точність при k-fold: {np.mean(fold_accuracies):.4f}")


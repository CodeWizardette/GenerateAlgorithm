
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = MLPClassifier(
    hidden_layer_sizes=(50, 25),  # İki gizli katman, sırasıyla 50 ve 25 nöron içeriyor
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    max_iter=1000,
    alpha=0.0001,  # L2 düzenlileştirme parametresi
    batch_size=32,  # Mini-batch boyutu
    random_state=42
)

model.fit(X_train_scaled, y_train)

y_train_pred = model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Eğitim Doğruluğu:", train_accuracy)

y_test_pred = model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Doğruluğu:", test_accuracy)
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

model = MLPClassifier(
    hidden_layer_sizes=(10,),  # Gizli katman boyutu (örneğin, 10)
    activation='relu',         # Aktivasyon fonksiyonu (örneğin, 'relu')
    solver='adam',              # Optimizasyon algoritması (örneğin, 'adam')
    learning_rate='constant',   # Öğrenme hızı (örneğin, 'constant')
    max_iter=1000               # Maksimum iterasyon sayısı (örneğin, 1000)
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

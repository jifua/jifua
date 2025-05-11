# Import Library
import numpy as np  # Untuk operasi numerik (array dan matriks)
import matplotlib.pyplot as plt  # Untuk visualisasi grafik
import tensorflow as tf  # Library deep learning
from tensorflow.keras.utils import plot_model  # Untuk visualisasi arsitektur model
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Untuk augmentasi gambar
from tensorflow.keras.applications import MobileNet  # Mengimpor arsitektur MobileNet
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization  # Layer-layer tambahan
from tensorflow.keras.models import Model  # Untuk membuat model kustom
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Untuk menghentikan training dan mengatur learning rate
from tensorflow.keras.optimizers import SGD  # Optimizer SGD
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, f1_score  # Evaluasi model
import seaborn as sns  # Visualisasi data (heatmap, dll.)
from tensorflow.keras.regularizers import l2  # Regularisasi L2 (menghindari overfitting)
from sklearn.preprocessing import label_binarize  # Konversi label ke bentuk biner (untuk ROC)

#Path Dataset
train_dir = '/content/drive/MyDrive/Tomato Leaf Disease/train'  # Path data training
val_dir = '/content/drive/MyDrive/Tomato Leaf Disease/val'  # Path data validasi
test_dir = '/content/drive/MyDrive/Tomato Leaf Disease/test'  # Path data testing

#Image Augmentation dan Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalisasi pixel (0-1)
    shear_range=0.2,  # Distorsi gambar
    zoom_range=0.2,  # Zoom in/out acak
    rotation_range=45,  # Rotasi acak
    width_shift_range=0.2,  # Geser horizontal
    height_shift_range=0.2,  # Geser vertikal
    horizontal_flip=True,  # Flip horizontal
    vertical_flip=True,  # Flip vertikal
    brightness_range=[0.8, 1.2],  # Variasi kecerahan
    fill_mode='nearest'  # Pengisian piksel kosong
)

val_datagen = ImageDataGenerator(rescale=1./255)  # Normalisasi data validasi
test_datagen = ImageDataGenerator(rescale=1./255)  # Normalisasi data testing

# Membuat Generator Gambar
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=16, class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(224, 224), batch_size=16, class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=16, class_mode='categorical', shuffle=False
)

#Load dan Modifikasi MobileNet
mobilenet = MobileNet(
    include_top=True, weights='imagenet', input_shape=(224, 224, 3),
    pooling=None, classes=1000, classifier_activation="softmax"
)

# Membekukan 10 layer pertama (tidak ikut training)
for layer in mobilenet.layers[:10]:
    layer.trainable = False

# Tambah Layer Kustom
x = Flatten(name='flatten_custom')(mobilenet.output)
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.3, name='dropout_1')(x)
x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.6, name='dropout_2')(x)
output = Dense(len(train_generator.class_indices), activation='softmax')(x)

# Membangun dan Menyusun Model
model_mobilenet = Model(inputs=mobilenet.input, outputs=output)
model_mobilenet.summary()  # Menampilkan struktur model
plot_model(model_mobilenet, to_file='model_mobilenet_architecture.png', show_shapes=True, show_layer_names=True)
print("Model summary telah disimpan sebagai 'model_mobilenet_architecture.png'.")

#Kompilasi Model
model_mobilenet.compile(optimizer=SGD(learning_rate=1e-4, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Callback untuk Menghentikan atau Menyesuaikan Training
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Training Model
history_mobilenet = model_mobilenet.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[early_stopping, reduce_lr]
)

#Evaluasi Model
test_loss_mobilenet, test_acc_mobilenet = model_mobilenet.evaluate(test_generator)
print(f'Test Accuracy: {test_acc_mobilenet * 100:.2f}%')
print(f'Test Loss: {test_loss_mobilenet :.4f}%')

# Visualisasi Akurasi & Loss
plt.figure(figsize=(14, 5))

# Akurasi
plt.subplot(1, 2, 1)
plt.plot(history_mobilenet.history['accuracy'], label='Train Accuracy')
plt.plot(history_mobilenet.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history_mobilenet.history['loss'], label='Train Loss')
plt.plot(history_mobilenet.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Confusion Matrix
y_true = test_generator.classes
y_pred = np.argmax(model_mobilenet.predict(test_generator), axis=-1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_generator.class_indices.keys(),
            yticklabels=test_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

#Classification Report & F1 Score
print('Classification Report:')
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

f1 = f1_score(y_true, y_pred, average='weighted')
print(f'F1 Score: {f1:.4f}')


#ROC Curve per Kelas
y_test_bin = label_binarize(y_true, classes=list(range(len(test_generator.class_indices))))
y_pred_prob = model_mobilenet.predict(test_generator)

plt.figure(figsize=(14, 10))
for i in range(len(test_generator.class_indices)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {list(test_generator.class_indices.keys())[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Garis referensi AUC=0.5
plt.title('ROC Curve for Each Class')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.show()


#Menyimpan Model
model_mobilenet.save('/content/drive/MyDrive/saved_model/model_mobilenet.h5')  # Simpan model ke Google Drive

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from flask import Flask, request, jsonify
import cv2
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
from PIL import Image
import io

app = Flask(__name__)

# Configuración
IMG_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 50
NUM_CLASSES = 13
K_FOLDS = 3  # Número de folds para validación cruzada

CLASSES = [
    "Huipil de Gala Istmeño Tradicional",
    "Huipil Cotidiano de Diario",
    "Enagua Formal de Celebración",
    "Enagua de Gala",
    "Rabona",
    "Huipil Ceremonial Religioso",
    "Sobrero de Charro Istmeño",
    "Guayabera",
    "Pantalón Tradicional del Hombre Istmeño",
    "Vestimenta de Luto",
    "Falda de Trazos",
    "Huipil de 3 Golpes",
    "Huipil cadenilla un golpe"
    "Otro"
]

# Información cultural correspondiente a cada clase
CULTURAL_INFO = {
    "Huipil de Gala Istmeño Tradicional": "Prenda ceremonial de gran valor cultural, elaborada con bordados florales intrincados que representan la biodiversidad del Istmo. Su confección puede tomar hasta 6 meses de trabajo artesanal.",
    "Huipil Cotidiano de Diario": "Versión más sencilla del huipil tradicional, usado para actividades diarias. Mantiene elementos decorativos pero con menor densidad de bordados y colores más sobrios.",
    "Enagua Formal de Celebración": "Falda amplia con olanes que forma parte del traje completo de tehuana. Para ocasiones especiales se elabora en terciopelo con bordados de hilo de oro.",
    "Enagua de Gala": "Conjunto completo usado durante las Velas y fiestas patronales. Destaca por su colorido y la abundancia de oro en los accesorios, simbolizando la riqueza cultural zapotecaLa enagua de gala es una prenda majestuosa, símbolo de identidad zapoteca y orgullo cultural en el Istmo. Es usada en eventos de gran relevancia como las velas istmeñas, bodas o la Guelaguetza. Su confección puede llevar semanas debido al detalle artesanal.",
    "Rabona": "La rabona surge como una adaptación práctica del traje tradicional istmeño. Es usada por mujeres zapotecas para el diario o eventos menos formales.",
    "Huipil Ceremonial Religioso": "Utilizado en ceremonias religiosas sincréticas, combina elementos católicos y zapotecas con simbolismos específicos según la festividad.",
    "Sombrero de Charro Istmeño": "Vestimenta masculina formal compuesta por guayabera bordada y pantalón negro. Representa la elegancia del hombre istmeño en eventos importantes.",
    "Guayabera": "Conjunto usado por varones durante celebraciones comunitarias, incluye camisa bordada, pantalón de manta y sombrero tradicional.",
    "Pantalón Tradicional del Hombre Istmeño": "El pantalón del hombre istmeño ha evolucionado a lo largo del tiempo, pasando de un diseño tradicional de fuerte raíz indígena, confeccionado artesanalmente con materiales naturales, a una versión más contemporánea, que mantiene la esencia de elegancia y sobriedad pero con cortes más modernos.",
    "Vestimenta de Luto": "Indumentaria negra usada durante periodos de duelo, con elementos simbólicos específicos que representan el tránsito a otra vida según cosmogonía zapoteca.",
    "Refajo Tradicional": "El refajo es una prenda interior tradicional que forma parte fundamental del vestuario femenino zapoteca. Aunque no siempre es visible, su presencia es crucial para dar volumen, estructura y elegancia al porte de la enagua exterior, especialmente en las presentaciones de gala o en actos culturales.",
    "Huipil de 3 Golpes": "Variante especial del huipil con tres secciones bordadas distintas que representan las tres etapas de la vida según la cosmovisión zapoteca.",
    "Huipil cadenilla un golpe":"Se caracteriza por su técnica de bordado conocida como cadenilla, realizada con una máquina de pedal y un solo movimiento continuo, lo que da origen al término un golpe",
    "Otro": "Vestimenta que no corresponde a ninguna de las categorías tradicionales específicas del Istmo de Oaxaca."
}

def create_model():
    """Crea un modelo CNN basado en MobileNetV2 con transfer learning"""
    # Usar MobileNetV2 como modelo base (eficiente para dispositivos móviles)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
    
    # Congelar las capas del modelo base
    for layer in base_model.layers:
        layer.trainable = False
    
    # Construir el modelo
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compilar el modelo
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_images_from_directory(directory, target_size):
    """Carga imágenes y etiquetas desde un directorio estructurado"""
    images = []
    labels = []
    class_indices = {}
    
    # Recorrer cada subdirectorio (clase)
    for i, class_name in enumerate(os.listdir(directory)):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path) and class_name in CLASSES:
            class_indices[class_name] = i
            # Recorrer cada imagen en la clase
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    # Cargar y preprocesar la imagen
                    img = tf.keras.preprocessing.image.load_img(
                        img_path, target_size=target_size)
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = preprocess_input(img_array)
                    
                    # Añadir a la lista
                    images.append(img_array)
                    labels.append(i)
                except Exception as e:
                    print(f"Error cargando {img_path}: {e}")
    
    return np.array(images), np.array(labels), class_indices

def prepare_cross_validation_data(dataset_dir):
    """Prepara los datos para validación cruzada"""
    print("Cargando imágenes para validación cruzada...")
    images, labels, class_indices = load_images_from_directory(dataset_dir, IMG_SIZE)
    
    # Convertir etiquetas a one-hot encoding
    labels_one_hot = tf.keras.utils.to_categorical(labels, NUM_CLASSES)
    
    return images, labels_one_hot, class_indices

def cross_validation_train(images, labels, n_folds=3):
    """Realiza entrenamiento con validación cruzada"""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_no = 1
    
    # Listas para almacenar resultados
    acc_per_fold = []
    loss_per_fold = []
    all_predictions = []
    all_true_labels = []
    
    # Dividir los datos en folds
    for train_idx, val_idx in kfold.split(images, np.argmax(labels, axis=1)):
        print(f"\nEntrenando fold {fold_no}/{n_folds}")
        
        # Dividir datos para este fold
        train_images = images[train_idx]
        train_labels = labels[train_idx]
        val_images = images[val_idx]
        val_labels = labels[val_idx]
        
        # Crear un modelo nuevo para cada fold
        model = create_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
        ]
        
        # Entrenar modelo
        history = model.fit(
            train_images, train_labels,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(val_images, val_labels),
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning
        # Descongelar últimas capas
        base_model = model.layers[0]
        for layer in base_model.layers[-30:]:
            layer.trainable = True
            
        # Recompilar con tasa de aprendizaje más baja
        model.compile(
            optimizer=Adam(learning_rate=0.00001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Entrenar con fine-tuning
        fine_history = model.fit(
            train_images, train_labels,
            epochs=20,
            batch_size=BATCH_SIZE,
            validation_data=(val_images, val_labels),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluar modelo
        scores = model.evaluate(val_images, val_labels, verbose=0)
        print(f"Fold {fold_no} - Loss: {scores[0]:.4f}, Accuracy: {scores[1]:.4f}")
        
        # Guardar resultados
        acc_per_fold.append(scores[1])
        loss_per_fold.append(scores[0])
        
        # Obtener predicciones
        y_pred = model.predict(val_images)
        all_predictions.extend(np.argmax(y_pred, axis=1))
        all_true_labels.extend(np.argmax(val_labels, axis=1))
        
        # Guardar modelo de este fold
        model.save(f'model_fold_{fold_no}.h5')
        
        fold_no += 1
    
    # Calcular promedios
    print("\nResultados de validación cruzada:")
    print(f"Accuracy promedio: {np.mean(acc_per_fold):.4f}")
    print(f"Loss promedio: {np.mean(loss_per_fold):.4f}")
    
    # Crear matriz de confusión general
    cm = confusion_matrix(all_true_labels, all_predictions)
    plot_confusion_matrix(cm, CLASSES, "confusion_matrix_all_folds.png")
    
    unique_labels = np.unique(all_true_labels)
    used_classes = [CLASSES[i] for i in unique_labels]
    # Informe de clasificación general
    cr = classification_report(all_true_labels, all_predictions, target_names=used_classes, output_dict=True)
    cr_df = pd.DataFrame(cr).transpose()
    print("\nInforme de clasificación general:")
    print(cr_df)
    
    # Guardar informe en CSV
    cr_df.to_csv("classification_report.csv")
    
    # Entrenar modelo final con todos los datos
    print("\nEntrenando modelo final con todos los datos...")
    final_model = create_model()
    
    final_model.fit(
        images, labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.00001)
        ],
        verbose=1
    )
    
    # Fine-tuning del modelo final
    base_model = final_model.layers[0]
    for layer in base_model.layers[-30:]:
        layer.trainable = True
        
    final_model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    final_model.fit(
        images, labels,
        epochs=20,
        batch_size=BATCH_SIZE,
        callbacks=[
            EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.000001)
        ],
        verbose=1
    )
    
    # Guardar modelo final
    final_model.save('fine_tuned_cultural_dress_model.h5')
    
    return final_model, acc_per_fold, loss_per_fold, cm, cr_df

def plot_confusion_matrix(cm, class_names, filename):
    """Crea una visualización de la matriz de confusión"""
    plt.figure(figsize=(12, 10))
    # Normalizar para obtener porcentajes
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Crear heatmap
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.title('Matriz de Confusión Normalizada')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    # También guardamos la matriz sin normalizar
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.title('Matriz de Confusión')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename.replace('.png', '_raw.png'))
    plt.close()

def prepare_data_generators(train_dir, val_dir, test_dir):
    """Prepara generadores de datos para entrenamiento tradicional"""
    # Aumentación de datos para entrenamiento
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Para validación y prueba solo preprocesamos
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    # Crear generadores
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def preprocess_image(image):
    """Preprocesa una imagen para la predicción"""
    img = cv2.resize(image, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
    img = preprocess_input(img)  # Preprocesar para MobileNetV2
    return np.expand_dims(img, axis=0)  # Añadir dimensión de lote

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de la API para predecir la clase de vestimenta"""
    try:
        # Recibir la imagen en formato base64
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No se proporcionó ninguna imagen'}), 400
        
        # Decodificar la imagen
        image_data = data['image']
        image_bytes = base64.b64decode(image_data)
        
        # Convertir a imagen CV2
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'No se pudo decodificar la imagen'}), 400
        
        # Preprocesar la imagen
        processed_img = preprocess_image(img)
        
        # Realizar la predicción
        model = load_model('fine_tuned_cultural_dress_model.h5')
        predictions = model.predict(processed_img)[0]
        
        # Obtener la clase con mayor probabilidad
        predicted_class_index = np.argmax(predictions)
        predicted_class = CLASSES[predicted_class_index]
        confidence = float(predictions[predicted_class_index])
        
        # Obtener información cultural
        cultural_info = CULTURAL_INFO[predicted_class]
        
        # Obtener las 3 principales predicciones
        top_indices = np.argsort(predictions)[-3:][::-1]
        top_predictions = [
            {
                'class': CLASSES[i],
                'confidence': float(predictions[i]),
                'cultural_info': CULTURAL_INFO[CLASSES[i]]
            }
            for i in top_indices
        ]
        
        # Devolver los resultados
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'cultural_info': cultural_info,
            'top_predictions': top_predictions
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def prepare_dataset_from_directory(dataset_dir):
    """Prepara un conjunto de datos dividido en entrenamiento, validación y prueba"""
    # Crear directorios necesarios
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    test_dir = os.path.join(dataset_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Crear subdirectorios para cada clase en cada directorio split
    for class_name in CLASSES:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
    return train_dir, val_dir, test_dir

if __name__ == '__main__':
    # Si se ejecuta directamente, entrenar el modelo
    import argparse
    
    parser = argparse.ArgumentParser(description='API y Entrenamiento de Red Neuronal para Clasificación de Vestimentas Tradicionales')
    parser.add_argument('--train', action='store_true', help='Entrenar el modelo')
    parser.add_argument('--cross-validation', action='store_true', help='Usar validación cruzada para entrenar')
    parser.add_argument('--dataset', type=str, default='dataset', help='Ruta al directorio del dataset')
    parser.add_argument('--serve', action='store_true', help='Iniciar la API')
    parser.add_argument('--port', type=int, default=5000, help='Puerto para la API')
    parser.add_argument('--folds', type=int, default=5, help='Número de folds para validación cruzada')
    
    args = parser.parse_args()
    
    if args.train:
        if args.cross_validation:
            print(f"Iniciando entrenamiento con validación cruzada de {args.folds} folds...")
            images, labels, class_indices = prepare_cross_validation_data(args.dataset)
            model, acc_per_fold, loss_per_fold, cm, report = cross_validation_train(
                images, labels, n_folds=args.folds)
            
            # Mostrar resultados
            print("\nResultados finales de validación cruzada:")
            print(f"Accuracy promedio: {np.mean(acc_per_fold):.4f}")
            print(f"Loss promedio: {np.mean(loss_per_fold):.4f}")
            print("\nPrecisión por fold:")
            for i, acc in enumerate(acc_per_fold):
                print(f"Fold {i+1}: {acc:.4f}")
        else:
            print("Preparando el conjunto de datos para entrenamiento tradicional...")
            train_dir, val_dir, test_dir = prepare_dataset_from_directory(args.dataset)
            
            print("Creando generadores de datos...")
            train_generator, val_generator, test_generator = prepare_data_generators(train_dir, val_dir, test_dir)
            
            print("Entrenando el modelo inicial...")
            model = create_model()
            
            # Entrenar el modelo inicial
            history = model.fit(
                train_generator,
                epochs=EPOCHS,
                validation_data=val_generator,
                callbacks=[
                    ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True),
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
                ]
            )
            
            print("Realizando fine-tuning...")
            # Descongelar últimas capas
            base_model = model.layers[0]
            for layer in base_model.layers[-30:]:
                layer.trainable = True
                
            # Recompilar con tasa de aprendizaje más baja
            model.compile(
                optimizer=Adam(learning_rate=0.00001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Entrenar con fine-tuning
            fine_history = model.fit(
                train_generator,
                epochs=20,
                validation_data=val_generator,
                callbacks=[
                    ModelCheckpoint('fine_tuned_model.h5', monitor='val_accuracy', save_best_only=True),
                    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.000001)
                ]
            )
            
            print("Evaluando el modelo...")
            # Evaluar modelo
            test_loss, test_acc = model.evaluate(test_generator)
            print(f"Test accuracy: {test_acc:.4f}")
            print(f"Test loss: {test_loss:.4f}")
            
            # Generar matriz de confusión
            predictions = model.predict(test_generator)
            y_pred = np.argmax(predictions, axis=1)
            y_true = test_generator.classes
            
            cm = confusion_matrix(y_true, y_pred)
            plot_confusion_matrix(cm, CLASSES, "confusion_matrix.png")
            
            # Generar informe de clasificación
            report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            print("\nInforme de clasificación:")
            print(report_df)
            report_df.to_csv("classification_report.csv")
            
            # Guardar el modelo final
            model.save('fine_tuned_cultural_dress_model.h5')
            
        print("¡Entrenamiento completado!")
    
    if args.serve:
        print(f"Iniciando API en puerto {args.port}...")
        # Asegurarse de que el modelo está cargado
        if not os.path.exists('fine_tuned_cultural_dress_model.h5'):
            print("No se encontró el modelo entrenado. Usando un modelo base.")
            model = create_model()
            model.save('fine_tuned_cultural_dress_model.h5')
        
        app.run(host='0.0.0.0', port=args.port)
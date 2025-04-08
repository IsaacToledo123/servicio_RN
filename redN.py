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
import logging

# Configuración del sistema de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Configuración
IMG_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 50
NUM_CLASSES = 14  # Actualizado para incluir todas las clases
K_FOLDS = 3  # Número de folds para validación cruzada

CLASSES = [
    "Enagua de Gala",
    "Enagua Formal de Celebración",
    "Guayabera",
    "Huipil cadenilla un golpe",
    "Huipil Ceremonial Religioso",
    "Huipil Cotidiano de Diario",
    "Huipil de 3 Golpes",
    "Huipil de Gala Istmeño Tradicional",
    "Pantalón Tradicional del Hombre Istmeño",
    "Rabona",
    "Refajo Tradicional",
    "Sobrero de Charro Istmeño",
    "Vestimenta de Luto",
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
    "Huipil cadenilla un golpe": "Se caracteriza por su técnica de bordado conocida como cadenilla, realizada con una máquina de pedal y un solo movimiento continuo, lo que da origen al término un golpe",
    "Otro": "Vestimenta que no corresponde a ninguna de las categorías tradicionales específicas del Istmo de Oaxaca."
}

def create_model():
    """Crea un modelo CNN basado en MobileNetV2 con transfer learning mejorado"""
    # Usar MobileNetV2 como modelo base con pesos pre-entrenados
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
    
    # Fine-tuning: Descongelamos las últimas capas desde el principio
    for layer in base_model.layers[:-50]:  # Congelar solo las primeras capas
        layer.trainable = False
    for layer in base_model.layers[-50:]:  # Descongelar las últimas capas
        layer.trainable = True
    
    # Construir un modelo más potente
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(1024, activation='relu'),  # Capa más grande
        Dropout(0.5),
        BatchNormalization(),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compilar el modelo con un optimizador más adecuado para fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Learning rate más bajo
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]  # Métricas adicionales
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
            logging.info(f"Cargando imágenes de la clase: {class_name}")
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
                    logging.error(f"Error cargando {img_path}: {e}")
    
    return np.array(images), np.array(labels), class_indices

def prepare_cross_validation_data(dataset_dir):
    """Prepara los datos para validación cruzada"""
    logging.info("Cargando imágenes para validación cruzada...")
    images, labels, class_indices = load_images_from_directory(dataset_dir, IMG_SIZE)
    
    # Convertir etiquetas a one-hot encoding
    labels_one_hot = tf.keras.utils.to_categorical(labels, NUM_CLASSES)
    
    logging.info(f"Datos cargados: {len(images)} imágenes, {len(np.unique(labels))} clases")
    
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
        logging.info(f"\nEntrenando fold {fold_no}/{n_folds}")
        
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
        for layer in base_model.layers[-50:]:  # Descongelar más capas
            layer.trainable = True
            
        # Recompilar con tasa de aprendizaje más baja
        model.compile(
            optimizer=Adam(learning_rate=0.00001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
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
        logging.info(f"Fold {fold_no} - Loss: {scores[0]:.4f}, Accuracy: {scores[1]:.4f}")
        
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
    logging.info("\nResultados de validación cruzada:")
    logging.info(f"Accuracy promedio: {np.mean(acc_per_fold):.4f}")
    logging.info(f"Loss promedio: {np.mean(loss_per_fold):.4f}")
    
    # Crear matriz de confusión general
    cm = confusion_matrix(all_true_labels, all_predictions)
    plot_confusion_matrix(cm, CLASSES, "confusion_matrix_all_folds.png")
    
    unique_labels = np.unique(all_true_labels)
    used_classes = [CLASSES[i] for i in unique_labels]
    # Informe de clasificación general
    cr = classification_report(all_true_labels, all_predictions, target_names=used_classes, output_dict=True)
    cr_df = pd.DataFrame(cr).transpose()
    logging.info("\nInforme de clasificación general:")
    logging.info(cr_df)
    
    # Guardar informe en CSV
    cr_df.to_csv("classification_report.csv")
    
    # Entrenar modelo final con todos los datos
    logging.info("\nEntrenando modelo final con todos los datos...")
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
    for layer in base_model.layers[-50:]:  # Descongelar más capas
        layer.trainable = True
        
    final_model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
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
    """Prepara generadores de datos para entrenamiento con augmentation mejorado"""
    # Aumentación de datos más intensa para entrenamiento
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,            # Más rotación
        width_shift_range=0.3,        # Más desplazamiento
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=False,          # No voltear verticalmente para prendas
        brightness_range=[0.7, 1.3],  # Variación de brillo
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
    # Verificar si la imagen es válida
    if image is None or image.size == 0:
        raise ValueError("Imagen inválida o vacía")
        
    # Asegurarse de que la imagen tenga el tamaño correcto
    img = cv2.resize(image, IMG_SIZE)
    
    # Asegurarse de que la imagen esté en RGB (OpenCV lee en BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalizar la imagen según lo requerido por MobileNetV2
    img = img.astype(np.float32)
    img = preprocess_input(img)
    
    # Añadir dimensión de lote
    img_batch = np.expand_dims(img, axis=0)
    
    return img_batch

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de la API para predecir la clase de vestimenta"""
    try:
        # Recibir la imagen en formato base64
        data = request.json
        if not data:
            logging.error("No se recibieron datos JSON")
            return jsonify({'error': 'No se recibieron datos JSON'}), 400
            
        if 'image' not in data:
            logging.error("No se proporcionó ninguna imagen en el JSON")
            return jsonify({'error': 'No se proporcionó ninguna imagen'}), 400
        
        # Decodificar la imagen
        try:
            image_data = data['image']
            # Verificar si la cadena base64 tiene el prefijo (data:image/jpeg;base64,)
            if ',' in image_data:
                image_data = image_data.split(',')[1]
                
            image_bytes = base64.b64decode(image_data)
            logging.info(f"Imagen decodificada: {len(image_bytes)} bytes")
        except Exception as e:
            logging.error(f"Error al decodificar base64: {str(e)}")
            return jsonify({'error': f'Error al decodificar la imagen: {str(e)}'}), 400
        
        # Convertir a imagen CV2
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logging.error("No se pudo convertir la imagen decodificada")
                return jsonify({'error': 'No se pudo decodificar la imagen'}), 400
                
            logging.info(f"Imagen cargada: {img.shape}")
        except Exception as e:
            logging.error(f"Error al procesar la imagen con OpenCV: {str(e)}")
            return jsonify({'error': f'Error al procesar la imagen: {str(e)}'}), 400
        
        # Preprocesar la imagen
        try:
            processed_img = preprocess_image(img)
            logging.info(f"Imagen preprocesada: {processed_img.shape}")
        except Exception as e:
            logging.error(f"Error al preprocesar la imagen: {str(e)}")
            return jsonify({'error': f'Error de preprocesamiento: {str(e)}'}), 400
        
        # Realizar la predicción
        try:
            model = load_model('fine_tuned_cultural_dress_model.h5')
            predictions = model.predict(processed_img)[0]
            logging.info(f"Predicciones obtenidas: {predictions}")
        except Exception as e:
            logging.error(f"Error al realizar la predicción: {str(e)}")
            return jsonify({'error': f'Error al realizar la predicción: {str(e)}'}), 500
        
        # Obtener la clase con mayor probabilidad
        predicted_class_index = np.argmax(predictions)
        if predicted_class_index >= len(CLASSES):
            logging.error(f"Índice de clase predicho fuera de rango: {predicted_class_index}")
            return jsonify({'error': 'Error en el modelo: índice de clase fuera de rango'}), 500
            
        predicted_class = CLASSES[predicted_class_index]
        confidence = float(predictions[predicted_class_index])
        
        # Obtener información cultural
        cultural_info = CULTURAL_INFO.get(predicted_class, "Información no disponible")
        
        # Obtener las 3 principales predicciones
        top_indices = np.argsort(predictions)[-3:][::-1]
        top_predictions = [
            {
                'class': CLASSES[i],
                'confidence': float(predictions[i]),
                'cultural_info': CULTURAL_INFO.get(CLASSES[i], "Información no disponible")
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
        
        logging.info(f"Predicción exitosa: {predicted_class} con confianza {confidence:.4f}")
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error no manejado: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({'error': f'Error interno del servidor: {str(e)}'}), 500

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

def test_api_locally():
    """Prueba la API localmente para verificar su funcionamiento"""
    try:
        # Cargar una imagen de prueba
        test_image_path = "path/to/test/image.jpg"  # Reemplaza con una ruta real
        if os.path.exists(test_image_path):
            with open(test_image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Simular una solicitud
            request_data = {"image": encoded_image}
            
            # Llamar directamente a la función de predicción
            from flask import Request
            with app.test_request_context(json=request_data):
                response = predict()
                print("Respuesta de prueba:", response)
        else:
            # Crear una imagen de prueba simple si no hay una imagen disponible
            test_img = np.zeros((160, 160, 3), dtype=np.uint8)
            # Añadir algún patrón simple para tener contenido
            cv2.rectangle(test_img, (20, 20), (140, 140), (255, 0, 0), 2)
            cv2.putText(test_img, "Test", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Codificar la imagen a base64
            _, buffer = cv2.imencode('.jpg', test_img)
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            
            # Simular una solicitud
            request_data = {"image": encoded_image}
            
            # Llamar directamente a la función de predicción
            from flask import Request
            with app.test_request_context(json=request_data):
                response = predict()
                print("Respuesta de prueba:", response)
            
        print("Prueba local exitosa")
    except Exception as e:
        print(f"Error en prueba local: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == '__main__':
    # Si se ejecuta directamente, entrenar el modelo
    import argparse
    
    parser = argparse.ArgumentParser(description='API y Entrenamiento de Red Neuronal para Clasificación de Vestimentas Tradicionales')
    parser.add_argument('--train', action='store_true', help='Entrenar el modelo')
    parser.add_argument('--cross-validation', action='store_true', help='Usar validación cruzada para entrenar')
    parser.add_argument('--dataset', type=str, default='dataset', help='Ruta al directorio del dataset')
    parser.add_argument('--serve', action='store_true', help='Iniciar la API')
    parser.add_argument('--test-api', action='store_true', help='Probar la API localmente')
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
            for layer in base_model.layers[-50:]:  # Más capas para fine-tuning
                layer.trainable = True
                
            # Recompilar con tasa de aprendizaje más baja
            model.compile(
                optimizer=Adam(learning_rate=0.00001),
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            
            # Continuar entrenamiento con fine-tuning
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
            
            # Evaluar en conjunto de prueba
            print("\nEvaluando en conjunto de prueba...")
            test_loss, test_acc, test_precision, test_recall = model.evaluate(test_generator)
            print(f"Test accuracy: {test_acc:.4f}")
            print(f"Test precision: {test_precision:.4f}")
            print(f"Test recall: {test_recall:.4f}")
            
            # Obtener predicciones en conjunto de prueba
            print("Generando predicciones en conjunto de prueba...")
            predictions = model.predict(test_generator)
            y_pred = np.argmax(predictions, axis=1)
            
            # Obtener etiquetas reales
            y_true = test_generator.classes
            
            # Crear matriz de confusión
            cm = confusion_matrix(y_true, y_pred)
            plot_confusion_matrix(cm, CLASSES, "confusion_matrix_test.png")
            
            # Informe de clasificación
            cr = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
            cr_df = pd.DataFrame(cr).transpose()
            print("\nInforme de clasificación en conjunto de prueba:")
            print(cr_df)
            
            # Guardar informe en CSV
            cr_df.to_csv("test_classification_report.csv")
            
            # Guardar modelo final
            model.save('fine_tuned_cultural_dress_model.h5')
            print("Modelo guardado como 'fine_tuned_cultural_dress_model.h5'")
            
    if args.test_api:
        print("Probando API localmente...")
        test_api_locally()
        
    if args.serve:
        print(f"Iniciando servidor API en puerto {args.port}...")
        # Verificar que el modelo existe
        if not os.path.exists('fine_tuned_cultural_dress_model.h5'):
            print("¡ADVERTENCIA! No se encontró el modelo entrenado.")
            print("Es posible que necesite entrenar primero con --train")
            
        # Configuración adicional para producción
        from waitress import serve
        print(f"Servidor escuchando en http://localhost:{args.port}")
        serve(app, host="0.0.0.0", port=args.port)
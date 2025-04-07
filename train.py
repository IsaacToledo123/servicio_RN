from redN import cross_validation_train, prepare_cross_validation_data, app
import tensorflow as tf
import numpy as np

# Configurar TensorFlow para crecimiento de memoria gradual
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

if __name__ == "__main__":
    # Ruta al directorio que contiene tus imágenes organizadas por clases
    dataset_dir = "dataset"
    
    # Modificar constantes en redN (si es posible acceder a ellas)
    # Si no se puede acceder directamente, modifica el archivo redN.py
    # Reducir el tamaño del batch
    import redN
    redN.BATCH_SIZE = 8  # Reducido de 32
    
    # Opcional: reducir el tamaño de las imágenes
    # redN.IMG_SIZE = (160, 160)  # Reducido de (224, 224)
    
    # Cargar imágenes y preparar datos
    images, labels, class_indices = prepare_cross_validation_data(dataset_dir)
    
    # Opcional: Limitar cantidad de imágenes para reducir consumo de memoria
    max_images = 1000  # Ajusta este número según tus restricciones de memoria
    if len(images) > max_images:
        indices = np.random.choice(len(images), max_images, replace=False)
        images = images[indices]
        labels = labels[indices]
    
    # Reducir número de folds
    n_folds = 3  # Reducido de 5
    
    print(f"Entrenando con {len(images)} imágenes y {n_folds} folds")
    
    # Iniciar entrenamiento con validación cruzada
    final_model, acc_per_fold, loss_per_fold, cm, cr_df = cross_validation_train(
        images, labels, n_folds=n_folds
    )
    
    # Si también quieres ejecutar la API después del entrenamiento
    app.run(debug=False, host='0.0.0.0', port=5000)
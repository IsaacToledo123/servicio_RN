from redN import cross_validation_train, prepare_cross_validation_data, app

if __name__ == "__main__":
    # Ruta al directorio que contiene tus imágenes organizadas por clases
    dataset_dir = "dataset"
    
    # Cargar imágenes y preparar datos
    images, labels, class_indices = prepare_cross_validation_data(dataset_dir)
    
    # Iniciar entrenamiento con validación cruzada
    final_model, acc_per_fold, loss_per_fold, cm, cr_df = cross_validation_train(
        images, labels, n_folds=5
    )
    
    # Si también quieres ejecutar la API después del entrenamiento
    # app.run(debug=False, host='0.0.0.0', port=5000)
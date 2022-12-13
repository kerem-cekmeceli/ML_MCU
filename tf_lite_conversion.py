import tensorflow as tf

def convert_to_tf_lite(train_set, model, tflite_model_name):
    train_set = train_set.numpy()
    # test_set = test_set.numpy()
    # train_labels = train_labels.numpy()
    # test_labels = test_labels.numpy()

    # Convert Keras model to a tflite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Convert the model to the TensorFlow Lite format with quantization
    quantize = True
    if (quantize):
        def representative_dataset():
            for i in range(500):
                shape = train_set[i].shape[0]
                yield([train_set[i].reshape(1,shape,13,1)])

        # Set the optimization flag.
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Enforce full-int8 quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8  quantization not compression 1/4
        converter.inference_output_type = tf.int8  # or tf.uint8
        # Provide a representative dataset to ensure we quantize correctly.
        
    converter.representative_dataset = representative_dataset
    tflite_model = converter.convert()

    open(tflite_model_name + '.tflite', 'wb').write(tflite_model)
    
    return tflite_model
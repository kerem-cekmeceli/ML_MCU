import tensorflow as tf
import os
import matplotlib.pyplot as plt

def convert_to_tf_lite(model, model_name, path_keras, path_tf_lite_nq, train_set=None, path_tf_lite_q=None):
    
    quantize = path_tf_lite_q is not None
    if quantize:
        assert train_set is not None

    # Convert the model to TFLite without quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model_nq = converter.convert()

    # Save the non-quantized TFLite model to disk
    nq_model_name_on_disk = "tf_lite_" + model_name + "_no_quant.tflite"
    open(path_tf_lite_nq + nq_model_name_on_disk, "wb").write(tflite_model_nq)
    
    # Quantization
    if quantize:
        def representative_dataset():
            for i in range(500):
                yield([train_set[i]])

        # Set the optimization flag.
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Enforce full-int8 quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8  quantization not compression 1/4
        converter.inference_output_type = tf.int8  # or tf.uint8
        # Provide a representative dataset to ensure we quantize correctly.
        converter.representative_dataset = representative_dataset
        tflite_model_q = converter.convert()
        
        # Save the non-quantized TFLite model to disk
        nq_model_name_on_disk = "tf_lite_" + model_name + "_no_quant.tflite"
        open(path_tf_lite_nq + nq_model_name_on_disk, "wb").write(tflite_model_q)
        
    sizes_on_disk = []
    comp_rat = []
    legends = []

    # Show the model size for the non-quantized HDF5 model
    h5_in_kb = os.path.getsize(path_keras + 'keras_' + model_name + '.h5') / 1024
    print("HDF5 Model size without quantization: %d KB" % h5_in_kb)
    sizes_on_disk.append(h5_in_kb)
    comp_rat.append(1.)
    legends.append("Keras")

    # Show the model size for the non-quantized TFLite model
    tflite_nq_in_kb = os.path.getsize(path_tf_lite_nq + nq_model_name_on_disk) / 1024
    print("TFLite Model size without quantization: %d KB" % tflite_nq_in_kb)
    sizes_on_disk.append(tflite_nq_in_kb)
    legends.append("TFLiteNonQuant")

    # Determine the reduction in model size
    comp = h5_in_kb / tflite_nq_in_kb
    print("\nReduction in model size from Keras to TFLite by a factor of %f" % (comp))
    comp_rat.append(comp)
    
    if quantize:
        q_model_name_on_disk = 'tf_lite_' + model_name + '_quant.tflite'
        # Save the non-quantized TFLite model to disk
        open(path_tf_lite_q + q_model_name_on_disk, 'wb').write(tflite_model_q)
        
        # Show the model size for the non-quantized TFLite model
        tflite_q_in_kb = os.path.getsize(path_tf_lite_q + q_model_name_on_disk) / 1024
        print("Quantized TFLite Model size: %d KB" % tflite_q_in_kb)
        sizes_on_disk.append(tflite_q_in_kb)
        legends.append("TFLiteQuant")
        
        # Determine the reduction in model size
        comp = h5_in_kb / tflite_q_in_kb
        print("\nReduction in model size from Keras to TFLiteQuantized by a factor of %f" % (comp))
        comp_rat.append(comp)
        
    # Plots for comparison
    plt.figure()
    plt.bar(legends, sizes_on_disk)
    plt.title("Sizes of the models")
    plt.grid()
    
    plt.figure()
    plt.bar(legends, comp_rat)
    plt.title("Compression Ratios")
    plt.grid()
    
        

def convert_to_tf_lite_quantized(model, model_name, path_tf_lite_q, train_set):
    train_set = train_set.numpy()

    # Convert Keras model to a tflite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Convert the model to the TensorFlow Lite format with quantization
    quantize = True
    if (quantize):
        def representative_dataset():
            for i in range(500):
                yield([train_set[i]])

        # Set the optimization flag.
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Enforce full-int8 quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8  quantization not compression 1/4
        converter.inference_output_type = tf.int8  # or tf.uint8
        # Provide a representative dataset to ensure we quantize correctly.
        
    converter.representative_dataset = representative_dataset
    tflite_model = converter.convert()

    open(path_tf_lite_q + 'tf_lite_' + model_name + '_quant.tflite', 'wb').write(tflite_model)
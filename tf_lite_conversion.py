import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

def convert_to_tf_lite(keras_model, path_keras_model, path_tf_lite_nq_model=None, 
                       train_set=None, path_tf_lite_q_model=None):
    
    save_quantized = path_tf_lite_q_model is not None
    save_non_quantized = path_tf_lite_nq_model is not None
    
    assert save_quantized or save_non_quantized
    assert path_keras_model.endswith(".h5")
   
    if save_quantized:
        assert train_set is not None
        assert path_tf_lite_nq_model.endswith(".tflite") 
        
    if save_non_quantized:
        assert path_tf_lite_q_model.endswith(".tflite") 
        

    # Convert the model to TFLite without quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model_nq = converter.convert()

    if save_non_quantized:
        # Save the non-quantized TFLite model to disk
        open(path_tf_lite_nq_model, "wb").write(tflite_model_nq)
    
    # Quantization
    if save_quantized: 
        def representative_dataset():
            for i in range(500):
                shape = train_set[i].shape[0]
                yield([tf.reshape(train_set[i], [1,shape,13,1])])

        # Set the optimization flag.
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Enforce full-int8 quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8  quantization not compression 1/4
        converter.inference_output_type = tf.int8  # or tf.uint8
        # Provide a representative dataset to ensure we quantize correctly.
        converter.representative_dataset = representative_dataset
        tflite_model_q = converter.convert()
        
        # Save the quantized TFLite model to disk
        open(path_tf_lite_q_model, "wb").write(tflite_model_q)
        
    sizes_on_disk = []
    comp_rat = []
    legends = []

    # Show the model size for the non-quantized HDF5 model
    h5_in_kb = os.path.getsize(path_keras_model) / 1024
    print("HDF5 Model size without quantization: %d KB" % h5_in_kb)
    sizes_on_disk.append(h5_in_kb)
    comp_rat.append(1.)
    legends.append("Keras")

    if save_non_quantized:
        # Show the model size for the non-quantized TFLite model
        tflite_nq_in_kb = os.path.getsize(path_tf_lite_nq_model) / 1024
        print("TFLite Model size without quantization: %d KB" % tflite_nq_in_kb)
        sizes_on_disk.append(tflite_nq_in_kb)
        legends.append("TFLiteNonQuant")

        # Determine the reduction in model size
        comp = h5_in_kb / tflite_nq_in_kb
        print("\nReduction in model size from Keras to TFLite by a factor of %f" % (comp))
        comp_rat.append(comp)
    
    if save_quantized:
        
        # Show the model size for the non-quantized TFLite model
        tflite_q_in_kb = os.path.getsize(path_tf_lite_q_model) / 1024
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
    plt.ylabel("KB")
    plt.title("Sizes of the models")
    plt.grid()
    
    plt.figure()
    plt.bar(legends, comp_rat)
    plt.title("Compression Ratios")
    plt.grid()
    
    if save_quantized and save_non_quantized:
        return tflite_model_nq, tflite_model_q
    elif save_non_quantized:
        return tflite_model_nq
    elif save_quantized:
        return tflite_model_q
    else: raise Exception
        

def eval_tf_lite_model(path_tf_lite_model, test_set, quantized):
    print("Evaluated model: ", path_tf_lite_model)
    
    # Test the model using the tflite interpreter with quantization
    interpreter = tf.lite.Interpreter(model_path=path_tf_lite_model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # print some information about the input and output tensors
    print("Input shape is {} and of type {}".format(input_details[0]['shape'], 
                                                    input_details[0]['dtype']))
    print("Output shape is {} and of type {}".format(output_details[0]['shape'], 
                                                     output_details[0]['dtype']))
    
    predictions = np.zeros((len(test_set),), dtype=int)
    
    if quantized:
        input_scale, input_zero_point = input_details[0]["quantization"]
        for i in range(len(test_set)):
            val_batch = test_set[i]
            val_batch = val_batch / input_scale + input_zero_point
            val_batch = np.expand_dims(val_batch, axis=0).astype(input_details[0]["dtype"])
            interpreter.set_tensor(input_details[0]['index'], val_batch)
            interpreter.allocate_tensors()
            interpreter.invoke()

            tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
            predictions[i] = tflite_model_predictions.argmax()
    else:
        for i in range(len(test_set)):
            val_batch = test_set[i]
            val_batch = np.expand_dims(val_batch, axis=0)
            interpreter.set_tensor(input_details[0]['index'], val_batch)
            interpreter.allocate_tensors()
            interpreter.invoke()
            tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
            predictions[i] = tflite_model_predictions.argmax()
        
    return predictions
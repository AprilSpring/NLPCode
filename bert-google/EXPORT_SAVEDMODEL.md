
# 如何生成savedmodel格式模型，并建立http端的请求（详见 bert/run_classifier_savedmodel_Vpredict.py）


## 1. 修改model_fn_builder()函数，为了使生成的模型结果（即output_spec）中有export_outputs变量。
```
去掉：
  '''
  output_spec = tf.contrib.tpu.TPUEstimatorSpec(
      mode=mode, predictions=probabilities, scaffold_fn=scaffold_fn)
  '''
  
添加：
  predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int32)
      export_outputs={'classes': tf.estimator.export.PredictOutput(
                            {"probabilities": probabilities, "classid": predictions})}
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, predictions=probabilities, scaffold_fn=scaffold_fn, export_outputs=export_outputs)
```

## 2. 添加 serving_input_fn()函数
```
 def serving_input_fn():
    label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
    input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='segment_ids')
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'label_ids': label_ids,
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
    })()
    return input_fn
```

## 3. 修改main()函数
```
  if FLAGS.do_export:
    estimator._export_to_tpu = False
    estimator.export_savedmodel(FLAGS.export_dir, serving_input_fn)
```

## 4. FLAGS处添加变量定义
```
  flags.DEFINE_string(
    "export_dir", None,
    "The dir where the exported model will be written.")

  flags.DEFINE_bool(
    "do_export", False,
    "Whether to export the model.")
```






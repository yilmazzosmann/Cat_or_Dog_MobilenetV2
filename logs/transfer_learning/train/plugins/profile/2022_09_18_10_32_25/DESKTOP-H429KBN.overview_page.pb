?  *	gfff???@2?
[Iterator::Model::MaxIntraOpParallelism::Prefetch::FiniteSkip::BatchV2::Shuffle::Zip[0]::Map ????\@!)0??W@)??6??\@1A:t4EW@:Preprocessing2|
EIterator::Model::MaxIntraOpParallelism::Prefetch::FiniteSkip::BatchV2y?&10_@!?????X@)?0?*?!@1???t?!@:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::FiniteSkip::BatchV2::Shuffle::Zip ?H?}?\@!2e^?#W@)P??n???1???f??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?E??????!y???????)?@??ǘ??1Q?yQU??:Preprocessing2?
cIterator::Model::MaxIntraOpParallelism::Prefetch::FiniteSkip::BatchV2::Shuffle::Zip[1]::TensorSlice ??y???!1c`Z???)??y???11c`Z???:Preprocessing2?
NIterator::Model::MaxIntraOpParallelism::Prefetch::FiniteSkip::BatchV2::Shuffle ?? ??\@!	(p??,W@)
ףp=
??1X??#rk??:Preprocessing2?
hIterator::Model::MaxIntraOpParallelism::Prefetch::FiniteSkip::BatchV2::Shuffle::Zip[0]::Map::TensorSlice =?U?????!? ??ꪺ?)=?U?????1? ??ꪺ?:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?N@aÓ?!?9?????)?N@aÓ?1?9?????:Preprocessing2s
<Iterator::Model::MaxIntraOpParallelism::Prefetch::FiniteSkip??:M0_@!?
5?.?X@)?J?4q?1?????k?:Preprocessing2F
Iterator::Model(~??k	??!L??v???)-C??6j?1ORG'!?d?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Y      Y@q?Ռ4???"?
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"GPU(: B??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.JDESKTOP-H429KBN: Failed to load libcupti (is it installed and accessible?)
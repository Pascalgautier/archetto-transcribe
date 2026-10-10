# MT3 (Google Magenta) sur CPU : usage .v/bin/python infer_mt3.py in.wav out.json  -> [[start, end, midi, velocity, program, is_drum], ...]
import sys, json, functools, os
import numpy as np, librosa, gin, jax, seqio, t5, t5x, tensorflow.compat.v2 as tf
import note_seq
from mt3 import metrics_utils, models, network, note_sequences, preprocessors, spectrograms, vocabularies
tf.config.set_visible_devices([], "GPU")
HERE = os.environ.get("MT3_HOME", os.path.dirname(os.path.abspath(__file__)))   # src/ (clone of magenta/mt3) + ckpt/checkpoints/mt3; SR = 16000

class InferenceModel:
    def __init__(self, ckpt, model_type="mt3"):
        num_velocity_bins = 1; self.encoding_spec = note_sequences.NoteEncodingWithTiesSpec; self.inputs_length = 256
        gin_files = [os.path.join(HERE, "src/mt3/gin/model.gin"), os.path.join(HERE, f"src/mt3/gin/{model_type}.gin")]
        self.batch_size = 8; self.outputs_length = 1024
        self.sequence_length = {"inputs": self.inputs_length, "targets": self.outputs_length}
        self.partitioner = t5x.partitioning.PjitPartitioner(num_partitions=1)
        self.spectrogram_config = spectrograms.SpectrogramConfig()
        self.codec = vocabularies.build_codec(vocab_config=vocabularies.VocabularyConfig(num_velocity_bins=num_velocity_bins))
        self.vocabulary = vocabularies.vocabulary_from_codec(self.codec)
        self.output_features = {"inputs": seqio.ContinuousFeature(dtype=tf.float32, rank=2), "targets": seqio.Feature(vocabulary=self.vocabulary)}
        with gin.unlock_config():
            gin.parse_config_files_and_bindings(gin_files, ["from __gin__ import dynamic_registration", "from mt3 import vocabularies",
                "VOCAB_CONFIG=@vocabularies.VocabularyConfig()", "vocabularies.VocabularyConfig.num_velocity_bins=%NUM_VELOCITY_BINS"], finalize_config=False)
        self.model = models.ContinuousInputsEncoderDecoderModel(module=network.Transformer(config=gin.get_configurable(network.T5Config)()),
            input_vocabulary=self.output_features["inputs"].vocabulary, output_vocabulary=self.output_features["targets"].vocabulary,
            optimizer_def=t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0), input_depth=spectrograms.input_depth(self.spectrogram_config))
        train_state_initializer = t5x.utils.TrainStateInitializer(optimizer_def=self.model.optimizer_def, init_fn=self.model.get_initial_variables,
            input_shapes={"encoder_input_tokens": (self.batch_size, self.inputs_length), "decoder_input_tokens": (self.batch_size, self.outputs_length)},
            partitioner=self.partitioner)
        restore = t5x.utils.RestoreCheckpointConfig(path=ckpt, mode="specific", dtype="float32")
        self._train_state = train_state_initializer.from_checkpoint_or_scratch([restore], init_rng=jax.random.PRNGKey(0))
        self._predict_fn = self.partitioner.partition(functools.partial(self.model.predict_batch_with_aux, num_decodes=1, rng=jax.random.PRNGKey(0)),
            in_axis_resources=(train_state_initializer.train_state_axes.params, t5x.partitioning.PartitionSpec("data")),
            out_axis_resources=t5x.partitioning.PartitionSpec("data"))
    def __call__(self, audio):
        ds = self.audio_to_dataset(audio); ds = self.preprocess(ds)
        model_ds = self.model.FEATURE_CONVERTER_CLS(pack=False)(ds, task_feature_lengths=self.sequence_length).batch(self.batch_size)
        preds = []
        for batch in model_ds.as_numpy_iterator():
            p, _ = self._predict_fn(self._train_state.params, batch); preds += list(self.vocabulary.decode_tf(p).numpy())
        preds = [{"est_tokens": t, "start_time": e["input_times"][0], "raw_inputs": []} for t, e in zip(preds, ds.as_numpy_iterator())]
        res = metrics_utils.event_predictions_to_ns(preds, codec=self.codec, encoding_spec=self.encoding_spec)
        return res["est_ns"]
    def audio_to_dataset(self, audio):
        frames, frame_times = self._audio_to_frames(audio)
        return tf.data.Dataset.from_tensors({"inputs": frames, "input_times": frame_times})
    def _audio_to_frames(self, audio):
        frame_size = self.spectrogram_config.hop_width
        padding = [0, frame_size - len(audio) % frame_size]; audio = np.pad(audio, padding, mode="constant")
        frames = spectrograms.split_audio(audio, self.spectrogram_config); num_frames = len(audio) // frame_size
        times = np.arange(num_frames) / self.spectrogram_config.frames_per_second
        return frames, times
    def preprocess(self, ds):
        pp = [functools.partial(t5.data.preprocessors.split_tokens_to_inputs_length, sequence_length=self.sequence_length, output_features=self.output_features, feature_key="inputs", additional_feature_keys=["input_times"]),
              preprocessors.add_dummy_targets, functools.partial(preprocessors.compute_spectrograms, spectrogram_config=self.spectrogram_config)]
        for f in pp: ds = f(ds)
        return ds

audio, _ = librosa.load(sys.argv[1], sr=SR, mono=True)
ns = InferenceModel(os.path.join(HERE, "ckpt/checkpoints/mt3"))(audio)
json.dump([[n.start_time, n.end_time, n.pitch, n.velocity, n.program, n.is_drum] for n in ns.notes], open(sys.argv[2], "w"))
print(len(ns.notes), "notes")

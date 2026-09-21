# Connection-test speech sample

`speech-sample.wav` is LibriSpeech utterance `1688-142285-0007`, converted from float PCM to 16 kHz mono signed 16-bit PCM WAV from the [PyTorch tutorial asset](https://download.pytorch.org/torchaudio/tutorial-assets/ctc-decoding/1688-142285-0007.wav). The complete utterance is preserved. Provenance: [PyTorch ASR tutorial](https://docs.pytorch.org/audio/2.1/tutorials/asr_inference_with_ctc_decoder_tutorial.html).

Attribution: Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur, *LibriSpeech: An ASR corpus based on public domain audio books*, 2015. Dataset: https://www.openslr.org/12/ . License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). No endorsement is implied.

The manifest records source and converted SHA-256 checksums. Read the float WAV with scipy.io.wavfile.read, then encode np.rint(np.clip(data,-1,1)*32767).astype('<i2') with wave.writeframes at 16000 Hz. No download or conversion occurs at runtime.

The English sample checks that a transcription response can be produced; it does not score recognition quality or establish support for other languages. Supply your own sample to test a configured language. No microphone is recorded during setup.

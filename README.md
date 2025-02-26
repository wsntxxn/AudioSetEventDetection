# Sound Event Detection on AudioSet

Training sound event detection models on [AudioSet](https://research.google.com/audioset/) and [AudioSet-strong](https://research.google.com/audioset/download_strong.html) data, mostly based on the wonderful [PANNs repository](https://github.com/qiuqiangkong/audioset_tagging_cnn). 

## Data Preparation

Preprocess AudioSet data to the required format in `data`:
```
train/
├── waveform.csv
├── waveform.h5
├── label_train_weak.csv
├── label_train.csv
├── label_val_weak.csv
├── label_val.csv
eval/
├── label_framed_posneg_weak.csv
├── label_framed_posneg.csv
├── waveform.csv
└── waveform.h5
```
The format of `label_weak.csv`, `label.csv`, `waveform.csv` and `waveform.h5` are provided in `example_data`.
You can process AudioSet files you maintain to follow this paradigm.
Some helper scripts are also provided in `data/*.py`.

## Training

```bash
python main.py train --config_file configs/cnn8_rnn_clip_frame.yaml
```
This will train Cnn8Rnn from scratch.
To train it with initialization of pre-trained checkpoints on AudioSet weak, you can refer to [PANNs repository](https://github.com/qiuqiangkong/audioset_tagging_cnn) to pre-train Cnn8Rnn.

## Evaluation
```bash
python main.py evaluate \
    --experiment_path experiments/clip_frame_loss/cnn8_rnn \
    --eval_config_file configs/audioset_eval.yaml
```

## Inference
```bash
python inference.py \
    --input xx.wav \
    --experiment_path experiments/clip_frame_loss/cnn8_rnn \
    --output prediction.csv
```

## Usage on Hugging Face

The pre-trained Cnn8Rnn is available on [Hugging Face](https://huggingface.co/wsntxxn/cnn8rnn-audioset-sed) for easy usage.
Feel free to use it.
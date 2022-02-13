### Dataset
340 000 image subset of MJSynth dataset.
https://drive.google.com/file/d/1AdDqu4j7RIxcg5ridRcr3qL74viJiX7B/view?usp=sharing


### Train

ViTSTR-Tiny without data augmentation 

```
python3 train.py \
--train_data /content/mjsynth_sample/train --valid_data /content/mjsynth_sample/test \
--valInterval=1000 --batch_size=192 --Transformation None --FeatureExtraction None  \
--SequenceModeling None --Prediction None --Transformer  --TransformerModel=vitstr_tiny_patch16_224 \
--imgH 224 --imgW 224  --manualSeed=69  --sensitive --workers=2
```

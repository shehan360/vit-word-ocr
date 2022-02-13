### Train

ViTSTR-Tiny without data augmentation 

```
python3 train.py \
--train_data /content/mjsynth_sample/train --valid_data /content/mjsynth_sample/test \
--valInterval=1000 --select_data MJ  --batch_ratio 0.5 --Transformation None --FeatureExtraction None  \
--SequenceModeling None --Prediction None --Transformer  --TransformerModel=vitstr_tiny_patch16_224 \
--imgH 224 --imgW 224  --manualSeed=69  --sensitive --workers=2
```

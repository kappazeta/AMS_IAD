# AMS: Ineligble Area Detection (IAD)
AMS: Ineligible Area Detection

## Requirements
`cm_fit` conda environment is used.

## Dataset
You can download data here: 


## Training 

You can create config file in `config` folder. To train, use the following command:
```
python pipeline.py --cfg config/train_0703.json --train
```
## Predicting

To predict, use the following command. It will create some eaxmple outputs in the `predicted_examples` directory.
```
python pipeline.py --cfg config/train_0703.json --predict
```

## Reproducing results
Basically, using config train_0703.json will give you the final results.

# QJ-plat
A platform consist of few-shot classification and object detection

## Installation
### Few-shot Recognition Environment
* Python 3
* Python packages
  - pytorch 1.0.0
  - torchvision 0.2.2
  - matplotlib
  - numpy
  - pillow
  - tensorboardX

An NVIDIA GPU and CUDA 9.0 or higher. 

### Open-world Object Detection Environment



### Open-vocabulary Object Detection Environment


### X-ray Environment


## Preparation
### Few-shot Recognition Dataset
You can download miniImagenet dataset from [here](https://drive.google.com/drive/folders/15WuREBvhEbSWo4fTr1r-vMY0C_6QWv4w).


You can download tieredImagenet dataset from [here](https://drive.google.com/file/d/1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07/view?usp=drive_open).


## Getting started
### Few-shot Recognition
Because WRN has a large amount of parameters. You can save the extracted feature before the classifaction layer to increase train or test speed. Here we provide the features extracted by WRN:
* miniImageNet: [train](https://drive.google.com/file/d/1uJ5-NhdDkdkqRhyrQoXKgkqoLt3BqWSC/view?usp=sharing), [val](https://drive.google.com/file/d/1p_6kalUR-a2so1yOGUn1DCAXL3ftgl-r/view?usp=sharing), [test](https://drive.google.com/file/d/1z69BN3ReZfSwpOt3P1l1LPDdqigKdsfT/view?usp=sharing)
* tieredImageNet: [train](https://drive.google.com/file/d/1Hz1Z4jVj8O3NQejUnpKeR9UTAVVdDw8T/view?usp=sharing), [val](https://drive.google.com/file/d/1DQ-LsyWtFsi6oyTxnBa5nQrla6lY7x0M/view?usp=sharing), [test](https://drive.google.com/file/d/1dGtfL8EEplJmiXGgxmQNtI36FYKyp-XG/view?usp=sharing)

You also can use our [pretrained WRN model](https://drive.google.com/drive/folders/1o51s2F7_bpG2k6JOgE9loYtSRIdOH2qc) to generate features for mini or tiered by yourself


## Training
### TRPN
```
# ************************** miniImagenet, 5way 1shot  *****************************
$ python3 conv4_train.py --dataset mini --num_ways 5 --num_shots 1 
$ python3 WRN_train.py --dataset mini --num_ways 5 --num_shots 1 

# ************************** miniImagenet, 5way 5shot *****************************
$ python3 conv4_train.py --dataset mini --num_ways 5 --num_shots 5 
$ python3 WRN_train.py --dataset mini --num_ways 5 --num_shots 5 

# ************************** tieredImagenet, 5way 1shot *****************************
$ python3 conv4_train.py --dataset tiered --num_ways 5 --num_shots 1 
$ python3 WRN_train.py --dataset tiered --num_ways 5 --num_shots 1 

# ************************** tieredImagenet, 5way 5shot *****************************
$ python3 conv4_train.py --dataset tiered --num_ways 5 --num_shots 5 
$ python3 WRN_train.py --dataset tiered --num_ways 5 --num_shots 5 

# **************** miniImagenet, 5way 5shot, 20% labeled (semi) *********************
$ python3 conv4_train.py --dataset mini --num_ways 5 --num_shots 5 --num_unlabeled 4

```

### IPN
```
# ************************** miniImagenet, 5way 1shot  *****************************
$ python3 train.py --dataset mini --num_ways 5 --num_shots 1 

# ************************** miniImagenet, 5way 5shot *****************************
$ python3 train.py --dataset mini --num_ways 5 --num_shots 5 

# ************************** tieredImagenet, 5way 1shot *****************************
$ python3 train.py --dataset tiered --num_ways 5 --num_shots 1 

# ************************** tieredImagenet, 5way 5shot *****************************
$ python3 train.py --dataset tiered --num_ways 5 --num_shots 5 

```


## Testing
### TRPN
``` 
# ************************** miniImagenet, Cway Kshot *****************************
$ python3 conv4_eval.py --test_model your_path --dataset mini --num_ways C --num_shots K 
$ python3 WRN_eval.py --test_model your_path --dataset mini --num_ways C --num_shots K 

```

### IPN
``` 
# ************************** miniImagenet, 5way 5shot *****************************
$ python3 eval.py --test_model your_path --dataset mini --num_ways C --num_shots K 

```

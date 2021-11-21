# Deep Covalent Site Identification (DeepCoSI)
  DeepCoSI: a Structure-based Deep Graph Learning Network Method for Covalent Binding Site Identification.
  
![error](https://github.com/Brian-hongyan/DeepCoSI/edit/main/deepcosi.tif)


# Environment
```
# software
chimera v1.13.1

# python
python 3.6.13

# environment to reproduce
conda env create -f DeepCoSI.yml
```


# DeepCoSI Training (An example)
```
nohup python -u ./codes/DeepCoSI_Train.py --gpuid 0 &
```
In order to speed up the calculation, we only used a small part of the data as an example.

# Covalent Site Identification 
We provide a well-trained model for users to predict the ligandability of cysteine.

```
# example
python ./codes/DeepCoSI_prediction.py 4hqr.pdb example
# Please check the prediction result in the build directory.
```

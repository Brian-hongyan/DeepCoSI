# Deep Covalent Site Identification (DeepCoSI)
  DeepCoSI: a Structure-based Deep Graph Learning Network Method for Covalent Binding Site Identification.
  
![error](https://github.com/Brian-hongyan/DeepCoSI/blob/main/deepcosi.jpg)


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
# A public dataset
We profiled the structures in RCSB PDB to identify potential cysteines for covalent ligand discovery. Please use it in http://cadd.zju.edu.cn/cidb/deepcosi/cys.

# Cite us
```
Hongyan Du Dejun Jiang Junbo Gao Xujun Zhang Lingxiao Jiang Yundian Zeng Zhenxing Wu Chao Shen Lei Xu Dongsheng Cao et al. Proteome-Wide Profiling of the Covalent-Druggable Cysteines with a Structure-Based Deep Graph Learning Network. Research. 2022;2022:DOI:10.34133/2022/9873564
```



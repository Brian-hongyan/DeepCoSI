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
@article{
doi:10.34133/2022/9873564,
author = {Hongyan Du  and Dejun Jiang  and Junbo Gao  and Xujun Zhang  and Lingxiao Jiang  and Yundian Zeng  and Zhenxing Wu  and Chao Shen  and Lei Xu  and Dongsheng Cao  and Tingjun Hou  and Peichen Pan },
title = {Proteome-Wide Profiling of the Covalent-Druggable Cysteines with a Structure-Based Deep Graph Learning Network},
journal = {Research},
volume = {2022},
number = {},
pages = {},
year = {2022},
doi = {10.34133/2022/9873564},
URL = {https://spj.science.org/doi/abs/10.34133/2022/9873564},
eprint = {https://spj.science.org/doi/pdf/10.34133/2022/9873564},
}
```



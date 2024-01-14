# Completely Occluded and Dense Object Instance Segmentation Using Box Prompt-Based Segmentation Foundation Models
Completely occluded and dense object instance segmentation (IS) is an important and challenging task. Although current amodal IS methods can predict invisible regions of occluded objects, they are difficult to deal with completely occluded objects. For dense object IS, existing box-based methods are overly dependent on the performance of bounding box detection. In this paper, we propose CFNet, a coarse-to-fine IS framework for completely occluded and dense objects, which is based on box prompt-based segmentation foundation models (BSMs). Specifically, CFNet first detects oriented bounding boxes (OBBs) that distinguish dense object instances and contour instances of occluded objects for coarse detection. Then, it generates a segmentation mask for each instance using a BSM for fine segmentation. For completely occluded objects, the detected contour instances will be converted into occluded object instances through geometric relationships, which overcomes the difficulty of directly predicting completely occluded object instances. Furthermore, using BSMs reduces the dependence on bounding box detection. Moreover, we propose a novel OBB prompt encoder and introduce a Gaussian label smoothing method for teacher targets in knowledge distillation. Experimental results demonstrate that CFNet achieves the best performance on both industrial and publicly available datasets.  
 
<div align=center>
<img src="https://github.com/zhen6618/OBBInstanceSegmentation/blob/master/Methods_Overview.png" width="1100px">
</div>

# Training
```
# train oriented bounding box detection module (ODM)
python OBB_Detection_Module/tools/train.py

# train OBB prompt-based segmentation module (OSM)
python OBB_Prompt_based_Segmentation_Module/OSM/train.py

# train OBB prompt-based segmentation module for knowledge distillation (OSM-KD)
python OBB_Prompt_based_Segmentation_Module/OSM_KD/train.py

```

# Inference
```
# test oriented bounding box detection module (ODM)
python OBB_Detection_Module/tools/test.py

# test OBB prompt-based segmentation module (OSM)
python OBB_Prompt_based_Segmentation_Module/OSM/inference.py

# test OBB prompt-based segmentation module for knowledge distillation (OSM-KD)
python OBB_Prompt_based_Segmentation_Module/OSM_KD/inference.py
```

# Citation




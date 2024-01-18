# Completely Occluded and Dense Object Instance Segmentation Using Box Prompt-Based Segmentation Foundation Models
<p align="justify">
Completely occluded and dense object instance segmentation (IS) is an important and challenging task. Although current amodal IS methods can predict invisible regions of occluded objects, they are difficult to deal with completely occluded objects. For dense object IS, existing box-based methods are overly dependent on the performance of bounding box detection. In this paper, we propose CFNet, a coarse-to-fine IS framework for completely occluded and dense objects, which is based on box prompt-based segmentation foundation models (BSMs). Specifically, CFNet first detects oriented bounding boxes (OBBs) that distinguish dense object instances and contour instances of occluded objects for coarse detection. Then, it generates a segmentation mask for each instance using a BSM for fine segmentation. For completely occluded objects, the detected contour instances will be converted into occluded object instances through geometric relationships, which overcomes the difficulty of directly predicting completely occluded object instances. Furthermore, using BSMs reduces the dependence on bounding box detection. Moreover, we propose a novel OBB prompt encoder and introduce a Gaussian label smoothing method for teacher targets in knowledge distillation. Experimental results demonstrate that CFNet achieves the best performance on both industrial and publicly available datasets.  
</p>

# Task
<div align=center>
<img src="https://github.com/zhen6618/OBBInstanceSegmentation/blob/master/figure/Task_Introduction.png" width="400px">
</div>
<p align="justify">
Examples of completely occluded and dense objects. Purple: unoccluded reference holes, yellow: horizontal bounding boxes that contain dense vehicles, green: oriented bounding boxes that contain dense vehicles or visible occluder (i.e., bolts or nuts) contours that are in the same planes as reference holes, orange: visible occluder contours that are in the same planes as reference holes, red: dense vehicles or occluders, blue: occluded reference holes.
</p>

# Method
1. CFNet 
<div align=center>
<img src="https://github.com/zhen6618/OBBInstanceSegmentation/blob/master/figure/Methods_Overview.png" width="1000px">
</div>
<p align="justify">
Architecture of the proposed CFNet. Compared with dense object IS, completely occluded object IS requires more post-processing steps to transform occluder instances into occludee instances. For clarity of presentation, only one instance is depicted in each result image of IS.
</p>

2. OBB Prompt Encoder
<div align=center>
<img src="https://github.com/zhen6618/OBBInstanceSegmentation/blob/master/figure/OBB_Prompt_Encoder.png" width="500px">
</div>
<p align="justify">
Architecture of the proposed OBB prompt encoder. The input is an OBB ($x, y, w, h, \theta$), where $(x, y)$, $w$, $h$ and $\theta$ represent the center point, width, height and orientation, respectively.
</p>

3. Knowledge Distillation on the OBB Prompt Encoder and Mask Decoder
<div align=center>
<img src="https://github.com/zhen6618/OBBInstanceSegmentation/blob/master/figure/Knowledge_Distillation.png" width="500px">
</div>
<p align="justify">
The process of KD for the OBB prompt encoder and mask decoder. ``TE``, ``BE`` and ``OE`` represent encoded feature embeddings with respect to the top-left point, bottom-right point and orientation of an OBB, respectively. ``GS`` stands for Gaussian smoothing.
</p>

# Robot System Design
<div align=center>
<img src="https://github.com/zhen6618/OBBInstanceSegmentation/blob/master/figure/Industrial_Dataset.png" width="600px">
</div>
<p align="justify">
Self-designed robotic vision system for completely occluded object IS in the industrial environment of the large commercial aircraft C919.
</p>

# Experiments
1. Completely Occluded Object IS
<div align=center>
<img src="https://github.com/zhen6618/OBBInstanceSegmentation/blob/master/figure/Completely_Occluded_Vis.png" width="900px">
      
<img src="https://github.com/zhen6618/OBBInstanceSegmentation/blob/master/figure/Completely_Occluded_Experiments.png" width="450px">
</div>

2. Dense Object IS
<div align=center>
<img src="https://github.com/zhen6618/OBBInstanceSegmentation/blob/master/figure/Dense_Vis.png" width="900px">
      
<img src="https://github.com/zhen6618/OBBInstanceSegmentation/blob/master/figure/Dense_Experiments.png" width="450px">
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
```
@InProceedings{zhou2024completely,
      title={Completely Occluded and Dense Object Instance Segmentation Using Box Prompt-Based Segmentation Foundation Models}, 
      author={Zhen Zhou and Junfeng Fan and Yunkai Ma and Sihan Zhao and Fengshui Jing and Min Tan},
      year={2024},
      booktitle={arXiv preprint arXiv:2401.08174},
}
```

# Acknowledgement
[lightning-sam](https://github.com/luca-medeiros/lightning-sam?tab=readme-ov-file)

[mmrotate](https://github.com/open-mmlab/mmrotate)

[segment-anything](https://github.com/facebookresearch/segment-anything)


# Occluded and Dense Object Instance Segmentation Using Box Prompt-Based Segmentation Foundation Models
<p align="justify">
Different from general instance segmentation (IS) tasks, occluded object IS in robot assembly and dense object IS in unmanned aerial vehicle (UAV) measurements are two more challenging robotic tasks. To uniformly deal with these difficulties, this paper proposes a unified coarse-to-fine IS framework, CFNet, which uses box prompt-based segmentation foundation models (BSMs). Specifically, CFNet first detects oriented bounding boxes (OBBs) to distinguish instances and provide coarse localization information. Then, it predicts OBB prompt-related masks for fine segmentation. CFNet performs IS on occluders and utilizes prior geometric properties to predict occluded object instances, which overcomes the difficulty of current amodal IS methods in directly predicting occluded objects. In addition, based on BSMs, CFNet alleviates the over-dependence on bounding box detection performance of existing IS methods using OBBs, improving dense object IS performance. Moreover, to enable BSMs to handle OBB prompts, we propose a novel OBB prompt encoder. To make CFNet more lightweight, we perform knowledge distillation on it and introduce a Gaussian label smoothing method for soft teacher targets. Experimental results demonstrate that CFNet outperforms all tested IS methods on both industrial and public datasets.
</p>

# Task
<div align=center>
<img src="https://github.com/zhen6618/OBBInstanceSegmentation/blob/master/figure/Task_Introduction.png" width="400px">
</div>
<p align="justify">
Examples of occluded and dense objects. Purple: unoccluded reference holes, yellow: horizontal bounding boxes that contain dense vehicles, green: oriented bounding boxes that contain dense vehicles or visible occluder (i.e., bolts or nuts) contours that are in the same planes as reference holes, orange: visible occluder contours that are in the same planes as reference holes, red: dense vehicles or occluders, blue: occluded reference holes.
</p>

# Method
1. CFNet 
<div align=center>
<img src="https://github.com/zhen6618/OBBInstanceSegmentation/blob/master/figure/Methods_Overview_.png" width="1000px"> 
</div>
<p align="justify">
Architecture of the proposed CFNet. Compared with dense object IS, occluded object IS requires more post-processing steps to transform occluder instances into occludee instances. For clarity of presentation, only one instance is depicted in each result image of IS.
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
Self-designed robotic system for occluded object IS in the industrial robot assembly environment of the large commercial aircraft C919.
</p>

# Experiments
1. Occluded Object IS
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
# Train OBB detection module (e.g., Oriented R-CNN with ResNet-18 as the backbone)
python OBB_Detection_Module/tools/train.py

# Train OBB prompt-based segmentation module (``OSM'' for short, we use it to train the teacher model)
python OBB_Prompt_based_Segmentation_Module/OSM/train.py

# Train OBB prompt-based segmentation module with knowledge distillation (``OSM_KD'' for short, we use it to train the student model)
python OBB_Prompt_based_Segmentation_Module/OSM_KD/train.py

```

# Inference
```
# Test oriented bounding box detection module (e.g., Oriented R-CNN with ResNet-18 as the backbone)
python OBB_Detection_Module/tools/test.py

# Test OBB prompt-based segmentation module (``OSM'' for short, we use it to test the teacher model)
python OBB_Prompt_based_Segmentation_Module/OSM/inference.py

# Test OBB prompt-based segmentation module with knowledge distillation (``OSM_KD'' for short, we use it to test the student model)
python OBB_Prompt_based_Segmentation_Module/OSM_KD/inference.py
```

# Citation
<!--
```
@InProceedings{zhou2024completely,
      title={Completely Occluded and Dense Object Instance Segmentation Using Box Prompt-Based Segmentation Foundation Models}, 
      author={Zhen Zhou and Junfeng Fan and Yunkai Ma and Sihan Zhao and Fengshui Jing and Min Tan},
      year={2024},
      booktitle={arXiv preprint arXiv:2401.08174},
}
```
-->

# Acknowledgement
[lightning-sam](https://github.com/luca-medeiros/lightning-sam?tab=readme-ov-file)

[mmrotate](https://github.com/open-mmlab/mmrotate)

[segment-anything](https://github.com/facebookresearch/segment-anything)


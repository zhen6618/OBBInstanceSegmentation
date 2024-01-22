## A Unified Instance Segmentation Framework for Occluded and Dense Objects in Robot Vision Measurement
<p align="justify">
Occluded and dense object instance segmentation in robot vision measurement are two challenging tasks. To uniformly deal with them, this paper proposes a unified coarse-to-fine instance segmentation framework, CFNet, which uses box prompt-based segmentation foundation models (BSMs), e.g., Segment Anything Model. Specifically, CFNet first detects oriented bounding boxes (OBBs) to distinguish instances and provide coarse localization information. Then, it predicts OBB prompt-related masks for fine segmentation. CFNet performs instance segmentation with partial OBBs on occluders to predict occluded object instances, which overcomes the difficulty of existing amodal instance segmentation methods in directly predicting occluded objects. In addition, since OBBs only serve as prompts, CFNet alleviates the over-dependence on bounding box detection performance of current instance segmentation methods using OBBs. Moreover, to enable BSMs to handle OBB prompts, we propose a novel OBB prompt encoder. To make CFNet more lightweight, we perform knowledge distillation on it and introduce a Gaussian label smoothing method for teacher model outputs. Experimental results demonstrate that CFNet outperforms all tested instance segmentation methods on both industrial and public datasets. 
</p>

# Task
<div align=center>
<img src="https://github.com/zhen6618/OBBInstanceSegmentation/blob/master/figure/Task_Introduction_.png" width="500px">
</div>
<p align="justify">
Examples of occluded (left) and dense (right) objects. Original objects are inside the corresponding black dotted boxes. Purple: unoccluded reference holes, yellow: horizontal bounding boxes that contain dense vehicles, green: oriented bounding boxes that contain dense vehicles or occluder (i.e., bolts or nuts) boundaries that are located at the contact surface between occluders and reference holes, orange: occluder boundaries that are located at the contact surface between occluders and reference holes, red: dense vehicles or occluders, blue: occluded reference holes.
</p>

# Method
1. CFNet 
<div align=center>
<img src="https://github.com/zhen6618/OBBInstanceSegmentation/blob/master/figure/Methods_Overview.png" width="1000px"> 
</div>
<p align="justify">
Architecture of the proposed CFNet. Compared with dense object instance segmentation, occluded object instance segmentation needs more post-processing steps to transform occluder instances into occludee instances.
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
The process of knowledge distillation for the OBB prompt encoder and mask decoder. ``TE``, ``BE`` and ``OE`` represent encoded feature embeddings with respect to the top-left point, bottom-right point and orientation of an OBB, respectively. ``GS`` stands for Gaussian smoothing.
</p>

# Robot System Design
<div align=center>
<img src="https://github.com/zhen6618/OBBInstanceSegmentation/blob/master/figure/Industrial_Dataset.png" width="600px">
</div>
<p align="justify">
Self-designed robotic system for occluded object instance segmentation in the industrial robot assembly environment of the large commercial aircraft C919.
</p>

# Experiments
1. Occluded Object Instance Segmentation
<div align=center>
<img src="https://github.com/zhen6618/OBBInstanceSegmentation/blob/master/figure/Completely_Occluded_Vis.png" width="900px">
      
<img src="https://github.com/zhen6618/OBBInstanceSegmentation/blob/master/figure/Completely_Occluded_Experiments.png" width="450px">
</div>

2. Dense Object Instance Segmentation
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


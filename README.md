# ContourNet: A general approach to detect contours of any type and shape for any object.
For example, it can be used to detect the unobstructed contour of a reference hole in complex industrial environments. First, the oriented bounding box detection module (ODM) is responsible for detecting the oriented bounding box (OBB) that obtains the unobstructed contour of the reference hole (see the green OBB). Second, the OBB prompt-based segmentation module (OSM) is used to segment the object that occludes the reference hole (see the red mask). Hence, the unobstructed contour of the reference hole is the contour of the object within the detected OBB. Futhermore, Knowledge distillation is applied to the mask decoder of OSM.
 
<div align=center>
<img src="https://github.com/zhen6618/ContourNet/blob/master/OBB_Prompt_based_Segmentation_Module/OSM/demo_pred_mask.png" width="300px">
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




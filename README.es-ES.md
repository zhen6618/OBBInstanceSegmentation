

## OBSeg: Marco de Segmentación de Instancias Preciso y Rápido que Utiliza Modelos de Segmentación Fundamentales con Prompts de Cajas Delimitadoras Orientadas
<p align="justify">
La segmentación precisa y rápida de instancias en imágenes de teledetección es un desafío de larga data. Dado que las cajas delimitadoras horizontales (HBB) introducen muchos objetos de interferencia, se suelen utilizar cajas delimitadoras orientadas (OBB) para la identificación de instancias. Sin embargo, bajo el paradigma de ``segmentación dentro de la caja delimitadora'', los métodos actuales de segmentación de instancias que utilizan OBB dependen en exceso del rendimiento de la detección de cajas delimitadoras. Recientemente, los modelos de segmentación fundamentales basados en prompts de caja (BSM, por sus siglas en inglés), como Segment Anything Model, se han desarrollado rápidamente y pueden mitigar esta dependencia. No obstante, los BSM existentes se basan en prompts de HBB, lo que no permite aprovechar plenamente sus capacidades. Para objetos con múltiples escalas, disposición densa y orientaciones arbitrarias, los prompts de HBB introducen muchas áreas de interferencia. Los métodos actuales que utilizan BSM con prompts de HBB, como RSPrompter, no pueden cumplir con los requisitos de segmentación de alta precisión. En este artículo, proponemos OBSeg, un marco de segmentación de instancias preciso y rápido que utiliza BSM con prompts de OBB. Específicamente, OBSeg detecta primero OBB para distinguir instancias y proporcionar información de localización aproximada. Luego, predice máscaras relacionadas con el prompt de OBB para una segmentación fina. Además, para habilitar que los BSM manejen prompts de OBB, proponemos un nuevo codificador de prompts OBB. Dado que las OBB solo funcionan como prompts, OBSeg reduce la dependencia excesiva en el rendimiento de la detección de cajas delimitadoras. Gracias a prompts de OBB más precisos, OBSeg supera a otros métodos de segmentación de instancias que utilizan BSM con prompts de HBB. Por otro lado, el equipamiento de teledetección, como los drones, tiene una necesidad más urgente de modelos ligeros. Para hacer que los BSM con prompts de OBB sean más ligeros, se introduce además un método de destilación de conocimiento basado en suavizado gaussiano con supervisión de objetivos de múltiples tipos. Los experimentos demuestran que OBSeg supera significativamente a los métodos actuales de segmentación de instancias en múltiples conjuntos de datos en términos de precisión de segmentación de instancias y presenta una velocidad de inferencia competitiva.
</p>

## Tarea
<div align=center>
<img src="https://github.com/zhen6618/OBBInstanceSegmentation/blob/master/figure/Task_Introduction.png" width="500px">
</div>
<p align="justify">
Para la segmentación de instancias en imágenes de teledetección, (a): las HBB introducen muchos objetos de interferencia. (b): El paradigma de ``segmentación dentro de la caja delimitadora'' limita la segmentación para que se realice principalmente dentro de la OBB detectada, haciendo que el rendimiento de la segmentación dependa en exceso del rendimiento de la detección de OBB. Una vez que la detección de OBB es inexacta, la segmentación de la máscara también se verá afectada. (c) OBSeg, el método propuesto, solo utiliza la OBB como un prompt para guiar la segmentación de objetos, por lo que el resultado de la segmentación depende menos del rendimiento de la detección de OBB. Aunque la detección de OBB sea inexacta, la máscara puede segmentarse con precisión.
</p>

## Método
1. OBSeg 
<div align=center>
<img src="https://github.com/zhen6618/OBBInstanceSegmentation/blob/master/figure/Methods_Overview_.png" width="1000px"> 
</div>
<p align="justify">
Arquitectura de OBSeg propuesta. Está compuesta principalmente por cuatro partes: un módulo de detección de OBB, un codificador de imagen, un codificador de prompts OBB y un decodificador de máscaras. OBSeg detecta primero OBB para distinguir instancias, identificar clases y proporcionar información de localización aproximada. Luego, el decodificador de máscaras utiliza los embeddings de imagen generados por el codificador de imagen y los embeddings de prompts OBB generados por el codificador de prompts OBB para generar máscaras de segmentación. Además, se aplica destilación de conocimiento basada en suavizado gaussiano con supervisión de objetivos de múltiples tipos al codificador de prompts OBB y al decodificador de máscaras para hacer OBSeg más ligero.
</p>

2. Codificador de Prompts OBB
<div align=center>
<img src="https://github.com/zhen6618/OBBInstanceSegmentation/blob/master/figure/OBB_Prompt_Encoder_.png" width="1000px">
</div>
<p align="justify">
Arquitectura del codificador de prompts OBB propuesto. La entrada es una OBB ($x, y, w, h, \theta$), donde $(x, y)$, $w$, $h$ y $\theta$ representan el punto central, el ancho, el alto y la orientación, respectivamente.
</p>

3. Destilación de Conocimiento en el Codificador de Prompts OBB y el Decodificador de Máscaras
<div align=center>
<img src="https://github.com/zhen6618/OBBInstanceSegmentation/blob/master/figure/Knowledge_Distillation_.png" width="850px">
</div>
<p align="justify">
Proceso de destilación de conocimiento para el codificador de prompts OBB y el decodificador de máscaras. ``TE``, ``BE`` y ``OE`` representan embeddings de características codificadas respecto al punto superior izquierdo, punto inferior derecho y orientación de una OBB, respectivamente. ``GS`` se refiere a suavizado gaussiano.
</p>

## Experimentos
<div align=center>
<img src="https://github.com/zhen6618/OBBInstanceSegmentation/blob/master/figure/Vis.png" width="900px">
</div>


## Instalación
```
pip install lightning
pip install pytorch
pip install opencv-python pycocotools matplotlib onnxruntime onnx
pip install -U openmim
mim install mmcv-full
mim install mmdet\<3.0.0
pip install mmrotate
```   

## Preparación de Tu Conjunto de Datos

## Entrenamiento
```
# Entrena el módulo de detección de OBB (p. ej., Oriented R-CNN con ResNet-18 como backbone)
python OBB_Detection_Module/tools/train.py

# Entrena el módulo de segmentación basado en prompts OBB (abreviado como ``OSM'', lo utilizamos para entrenar el modelo maestro)
python OBB_Prompt_based_Segmentation_Module/OSM/train.py

# Entrena el módulo de segmentación basado en prompts OBB con destilación de conocimiento (abreviado como ``OSM_KD'', lo utilizamos para entrenar el modelo estudiante)
python OBB_Prompt_based_Segmentation_Module/OSM_KD/train.py

```

## Inferencia
```
# Prueba el módulo de detección de cajas delimitadoras orientadas (p. ej., Oriented R-CNN con ResNet-18 como backbone)
python OBB_Detection_Module/tools/test.py

# Prueba el módulo de segmentación basado en prompts OBB (abreviado como ``OSM'', lo utilizamos para probar el modelo maestro)
python OBB_Prompt_based_Segmentation_Module/OSM/inference.py

# Prueba el módulo de segmentación basado en prompts OBB con destilación de conocimiento (abreviado como ``OSM_KD'', lo utilizamos para probar el modelo estudiante)
python OBB_Prompt_based_Segmentation_Module/OSM_KD/inference.py
```

## Cita
<!--
```
@InProceedings{zhou2024efficientinstancesegmentationframework,
      title={An Efficient Instance Segmentation Framework Based on Oriented Bounding Boxes}, 
      author={Zhen Zhou and Junfeng Fan and Yunkai Ma and Sihan Zhao and Fengshui Jing and Min Tan},
      year={2024},
      booktitle={arXiv preprint arXiv:2401.08174},
}
```
-->

## Agradecimientos
[lightning-sam](https://github.com/luca-medeiros/lightning-sam?tab=readme-ov-file)

[mmrotate](https://github.com/open-mmlab/mmrotate)

[segment-anything](https://github.com/facebookresearch/segment-anything)

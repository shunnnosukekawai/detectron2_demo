Detectron2 simple demo
====

You can get the resulting image and the output jsonfile easily.
The jsonfile contains the size of the input image and the information of the detected boxes.
In addition, you need to set a model in the model directory, and to use the following command in the detectron2_demo directory:
```
python demo/main.py -i <input file path> -o <output file path>
```
---

## Sample results from detectron2



* detected output image

| <img src="output/image1.jpg" width=400> | <img src="output/image2.jpg" width=400> |
|---------------------------------------------------------------------------|---------------------------------------------------------------------------|

* output jsonfile

<img src="output/outputjson_sample.jpg" width=400>




---

## explanation of json keys


   * inputImgShape  : size of input image (height, width)
   * detectedBoxes  : information of detected boxes
     * category  : category of detected box
     * detectedBoxArea  : (x1, y1, x2, y2)
     * overallRatioOfDetectedBox  : (x1/inputImgShape's width, y1/inputImgShape's height ,x2/inputImgShape's width , y2/inputImgShape's height)
     * confidenceScore  : conficence score of detected box

---
## Selectable models  
set one of these models in the model directory:

 | Architecture                                                                                                  | No. images | AP     | AP50   | AP75   | AP Small | AP Medium | AP Large | Model size full | Model size trimmed |
 |---------------------------------------------------------------------------------------------------------------|------------|--------|--------|--------|----------|-----------|----------|--------------------|-----------------|
 | [MaskRCNN Resnext101_32x8d FPN 3X](https://www.dropbox.com/sh/1098ym6vhad4zi6/AABe16eSdY_34KGp52W0ruwha?dl=0) | 191,832    | 90.574 | 97.704 | 95.555 | 39.904   | 76.350    | 95.165   | 816M               | 410M            |
 | [MaskRCNN Resnet101 FPN 3X](https://www.dropbox.com/sh/wgt9skz67usliei/AAD9n6qbsyMz1Y3CwpZpHXCpa?dl=0)        | 191,832    | 90.335 | 96.900 | 94.609 | 36.588   | 73.672    | 94.533   |480M                    | 240M            |
 | [MaskRCNN Resnet50 FPN 3X](https://www.dropbox.com/sh/44ez171b2qaocd2/AAB0huidzzOXeo99QdplZRjua?dl=0)                                                                                                              | 191,832           | 87.219       | 96.949       | 94.385       | 38.164         | 72.292          |  94.081        |                    |  168M               |

---

## Requirement
you can set the following requirements by using the command: `poetry install`

- Python = 3.6
- PyTorch = 1.4.0
- OpenCV >=4.2.0.32
- torchvision >=0.5.0
- click = 7.1.1

Activate the virtualenv by using the command:`poetry shell`
Install pycocotools by using this pip command:
- pycocotools: `pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`

After getting the above depencencies, run:
- detectron2: `python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`

After setting the above requirements, you can use this command to get the results:`python demo/main.py -i <input file path> -o <output file path>`

## Licence

Detectron2 is released under the [Apache 2.0 license](LICENSE).

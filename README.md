# cv-bicycle-detection
Bicycle Detection using Mask RCNN for the CV by Deep Learning course at TU Delft

You can run the project using the included notebook. It is advised to run this in Google Colab and all necessary libraries, datasets and imports are handled within the notebook.

## Generating the DelftBikes augmented dataset

In order to generate a new dataset you run:

```
 cd src
 python gen_data.py --model /PATH/TO/COCO/WEIGHTS --input /PATH/TO/DELFTBIKES  \
    --coco /PATH/TO/COCO --output /PATH/TO/OUTPUTIMAGE --ann /PATH/TO/OUTPUTANNOTATION
```

For convenience the notebook retrieves a generated dataset from a shared drive.

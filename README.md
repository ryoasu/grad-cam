# grad-cam
Grad-CAM (Gradient-weighted Class Activation Mapping)

The paper: https://arxiv.org/pdf/1610.02391v1.pdf

**Input images to imagenet**

<img src=./example/images/chimpanzee.png width=200> <img src=./example/images/elephant.png width=200> <img src=./example/images/lesser_panda.png width=200> <img src=./example/images/macaw.png width=200>

**Grad-CAM output images**

<img src=./assets/grad_cam-vgg16-chimpanzee.png width=200> <img src=./assets/grad_cam-vgg16-elephant.png width=200> <img src=./assets/grad_cam-vgg16-lesser_panda.png width=200> <img src=./assets/grad_cam-vgg16-macaw.png width=200>

## Requirement
- Python (3.6.3)
    - Keras (2.0.9)
    - numpy (1.14.0)
    - tensorflow (1.4.1)
    - opencv-python (3.4.0.12)
    - Pillow (5.0.0)

## Usage
1. Create config yaml file ([details about config](README.md/grad-cam#config-yaml)) 
2. Run grad-cam
  ```
  > python3 grad_cam path/to/config.yml
  ```

### Config (yaml)
```yaml
# <Use Library name>
keras:
  model:
    # <[optional] path to model architecture file (e.g. keras: *.json or *.yml)>
    architecture: path/to/architecture_file.yml
    # <It is also possible to load model architecture from source file.>
    # source:
    #   path: ./example/src/vgg16.py
    #   definition: vgg16
    #   args:
    #     - [224, 224] # image_size
    #     - 3          # channel
    #     - 1000       # classes
    # <path to model params (weight) file (e.g. keras: *.h5)>
    params: path/to/params_file.h5
    # <target layer name for grad-cam>
    layer: layer_name
  image:
    # <path to image (*.jpg or *.png)>
    path: path/to/image.png
    # <If you want to target multiple images, specify the image directory.>
    # path: path/to/image_dir
    # <path to output dir>
    output: path/to/output_dir
    # <[optional] image preprocessing>
    source:
      # <path to image preprocessing module (source file)>
      path: path/to/preprocessing.py
      # <definition (defined in image preprocessing source file>)
      definition: definition_name

```

### Image pre-processing
If image pre-processing is necessary, it is also possible to define the pre-processing module.
- Input
    - `path:str`: path to image file (Determined by cofig yaml)
    - `shape`: image shape (determined by model **input shape**)
- output
    - `image_array:ndarray`: ndarray whose shape is **input shape**

#### pre-processing module example for imagenet (python/keras)
```python
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import numpy as np

def image_to_arr(path, shape):
    img = image.load_img(path, target_size=shape[0:2])
    x = image.img_to_array(img)
    x = preprocess_input(x)
    return x
```

### When model architecture is defined in source code
You can load the model architecture from a `class` or `user-defined function` (python source code) in which the model is defined.

#### usage
Specify the class name or user-defined function name in which source file and model are defined in config (yaml).

※ 
`source code` and `architecutre file` can not be specified at the same time.
```yaml
keras:
  model:
    source:
      # path source code
      path: ./example/src/vgg16.py
      # name of the class or user-defined function
      definition: definition_name
      args:
        - arg1 # first argument of definition
        - arg2 # second argument of definition
        - arg3 # third argument of definition
    params: ./example/model/vgg16_weights_tf_dim_ordering_tf_kernels.h5
    layer: block5_conv3
```


examples of model defined [python source code](example/src/vgg16.py)



## Example
1. Download model params file for ***imagenet***
    ```sh
    # ./grad-cam
    wget  https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5 -P ./example/model
    ```
2. Run grad-cam
    ```sh
    # ./grad-cam
    python3 grad_cam.py ./example/config.yml
    ```

## References
- Torch implementation by the paper authors: https://github.com/ramprs/grad-cam
- Keras implementation: https://github.com/jacobgil/keras-grad-cam
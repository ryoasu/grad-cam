# grad-cam
Grad-CAM (Gradient-weighted Class Activation Mapping)

The paper: https://arxiv.org/pdf/1610.02391v1.pdf

## Requirement
- Python (3.6.3)
    - Keras (2.0.9)
    - numpy (1.14.0)
    - tensorflow (1.4.1)
    - opencv-python (3.4.0.12)
    - Pillow (5.0.0)

## Usage
`python3 grad_cam path/to/config.yml`

### Config (yaml)
```yaml
# Use Library name
keras:
  target:
    # [optional] path to model architecture file (e.g. keras: *.json or *.yml)
    architecture: path/to/architecture_file.yml
    # path to model params (weight) file (e.g. keras: *.h5)
    params: path/to/params_file.h5
    # target layer name for grad-cam
    layer: conv_layer
    image:
      # path to image (*.jpg or *.png)
      path: path/to/image.jpg
    # [optional] image preprocessing
    preprocessing:
      # path to image preprocessing module (source file)
      source: path/to/preprocessing.py
      # function name (defined in image preprocessing source file)
      function: image_to_arr

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

## Example
1. Download model params file for ***imagenet***
    ```sh
    # ./grad-cam
    wget  https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5 -P ./example/model
    ```
3. Modify `./example/config.yml` (default image is `./example/images/chimpanzee.png`)
2. Run grad-cam
    ```sh
    # ./grad-cam
    python3 grad_cam ./example/config.yml
    ```
**Output grad-cam image**

<img src=./assets/grad_cam-vgg16-chimpanzee.png width=200>
<img src=./assets/grad_cam-vgg16-elephant.png width=200>
<img src=./assets/grad_cam-vgg16-lesser_panda.png width=200>
<img src=./assets/grad_cam-vgg16-macaw.png width=200>


## References
- Torch implementation by the paper authors: https://github.com/ramprs/grad-cam
- Keras implementation: https://github.com/jacobgil/keras-grad-cam
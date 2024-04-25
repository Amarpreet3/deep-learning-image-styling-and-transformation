![Python 2.7](https://img.shields.io/badge/python-2.7-green.svg)
![Python 3.5](https://img.shields.io/badge/python-3.5-green.svg)

# Image Styling and Transformation
**Author: Amarpreet Kaur** | 📧 [Email](mailto:amarpreet.kaur@torontomu.ca) | 🏫 Toronto Metropolitan University

## Project Description
This project introduces a novel framework for photorealistic style transfer, designed to significantly enhance image styling and transformation across multiple domains, including interior design, fashion, and digital marketing. By leveraging an innovative autoencoder architecture with block training, high-frequency residual skip connections, and bottleneck feature aggregation, this framework excels in separating and recombining the content and style of arbitrary images, leading to superior styling outcomes.

# Style transfer: 



![output_7_4](https://github.com/Amarpreet3/Deep-Learning-Image-Styling-and-Transformation/assets/96805692/62be244b-4777-4fa7-a7d1-c98999e7a417)

## Key Features
- **Efficient Photorealistic Style Transfer:** Achieves high-quality image transformations.
- **Real-time Application:** Supports dynamic adjustments suitable for virtual environments and interactive workflows.
- **Advanced Neural Architecture:** Utilizes a novel combination of neural network strategies for enhanced style transfer capabilities.

  ## Repository Structure
Here's a breakdown of the main directories and files in this repository:

- **`.ipynb_checkpoints`**: Stores checkpoint files.
- **`banner`**: Contains banner images or graphics used in documentation.
- **`ckpts`**: Includes model checkpoints from training sessions, allowing model restoration or reuse.
- **`dataset`**: Directory for dataset, including links for both MSCOCO and ADE20K datasets, subdivided by training, validation, and test sets.
- **`figures`**: Holds example images for testing and demonstration purposes. This includes content and style images used in style transfer demonstrations.
- **`final_add_3_decoder`**: Contains scripts and models specific to the `add_3_decoder` architecture variant used for advanced decoding tasks.
- **`results`**: Output directory where stylized images are saved after processing.
- **`resultsWCT(baseline)`**: Stores baseline results using the Whitening and Coloring Transform (WCT) technique for comparison with the new method.
- **`utils`**: Utility scripts including support for image processing, model definitions, and other helper functions.
- **`README.md`**: The main documentation file providing an overview and instructions for using this repository.
- **`relu_demo.py`**: A demonstration script that shows how to apply the style transfer using the ReLU model configurations.
- **`test.py`**: Script for testing the models with different configurations and datasets.
- **`train.py`**: Contains the training code for the autoencoder, detailing setup, execution, and options for various training regimes.
## Datasets

### MSCOCO Dataset
To train my autoencoder, I utilize the MSCOCO dataset, which has been widely used in various style transfer studies including some WCT (Whitening and Coloring Transform) papers. The dataset consists of:
**Training Set**: 118,288 images,  **Validation Set**: 5,000 images and  **Test Set**: 40,670 images

[Download MSCOCO Dataset](https://cocodataset.org/#download)

### ADE20K Dataset
For my initial semantic segmentation approach, I employ the ADE20K dataset. This dataset is instrumental in our experiments, featuring:
**Training Set**: 25,574 images and **Validation Set**: 2,000 images


[Download ADE20K Dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/)

## Models and files
I applied BFA to the following model:
A pre-trained VGG-19 encoder (from input layer to **relu_4_1** layer; fixed during training) and a blockwisely trained decoder which can reproduce the **relu_3_1**, **relu_2_1**, **relu_1_1** features and the input image. The ZCA transformations are embedded at the bottleneck and the reproduced reluN1 layers in the decoder.
    - The model is in ```utils/model_relu.py``` and the associated checkpoint is in ```ckpts/ckpts-relu```.
    - A demo that uses this model to stylize example images in ```figures/``` is shown in ```relu_demo.py```. The resulting stylized images are in ```results/```.

Stylization with both models requires guided filtering in ```utils/photo_gif.py``` as the post-processing.



## Stylize images
To stylize a image put the content images in ```figures/content``` and the style images in ```figure/style``` then run ```relu_demo.py```

### Running the Demo
1. Add content images to `figures/content`.
2. Add style images to `figures/style`.
3. Execute:
   ```bash
   python relu_demo.py
   ```

## Training
```train.py``` is the training code for our model. The usage is provided in the file.
```
Detailed usage instructions are inside the script.

## Requirements 
- tensorflow v2.0.0 or above (I developed the models with tf-v2.4.1 and I also tested them in tf-v2.0.0)
- Python 3.x
- keras 2.0.x
- scikit-image
  
### Tutorial

Please check out the [FastPhotoStyle Tutorial](https://github.com/NVIDIA/FastPhotoStyle/blob/master/TUTORIAL.md).

## Citation

This README is designed to be straightforward, informative, and easy to navigate, providing all necessary details to understand, use, and contribute to the project effectively.


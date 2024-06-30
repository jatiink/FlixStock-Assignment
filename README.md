# FlixStock-Assignment
## Image Processing and Multithreading Solutions

This repository contains solutions to three problem statements involving image processing and multithreading in Python.

## Problem Statement 1: Create Image

### Task
Given the image `input.jpg` and mask `mask.png`, create the image `result.jpg`.

### Deliverables
- Code in Python
- Resulting image

### Usage
1. Place `input.jpg` and `mask.png` in the `images` folder.
2. Run the script:
```
python create_image.py
```
3. The resulting image `result.jpg` will be saved in the `images` folder.

### Example Solution
| Input | Mask | Result |
| ----- | ---- | ------ |
| <img src="https://github.com/jatiink/FlixStock-Assignment/assets/97089717/6effe06c-ac16-4094-bcbf-77415b9961f8" alt="drawing" width="300"/> | <img src="https://github.com/jatiink/FlixStock-Assignment/assets/97089717/724f07bf-e0d2-4e36-9cea-88a01666e432" alt="drawing" width="300"/> |  <img src="https://github.com/jatiink/FlixStock-Assignment/assets/97089717/9d28c99b-93f0-4dc2-a067-f6484bdccaa1" alt="drawing" width="300"/> | 



## Problem Statement 2: MultiThreading

### Task
Write a Python code to launch 3 different threads with the following behavior:
- Each thread should print every 5 seconds: `Thread <thread number> is running at <time elapsed>`.
- Initially start thread 1 and 3.
- After 20 seconds, stop thread 1 and start thread 2.
- After another 18 seconds, stop thread 3 and start thread 1.

### Deliverables
- Code in Python

### Usage
1. Run the script:
```
python multithreading.py
```

## Problem Statement 3: Similar Image

### Task
Given a query image (`query.jpg`) and a database folder of jpg images, find the image in the database folder that is most visually similar to the query image.

### Deliverables
- Code in Python
- Most similar image to query image

### Usage
1. Install the required libraries: Ensure you have all the necessary Python libraries installed. You can install them using pip: 
```
pip install torch torchvision pillow numpy scikit-learn matplotlib
```
2. Save the script: Save the provided code into a Python file, for example, similar_image_search.py.
3. Prepare your dataset: Ensure you have a folder with images for creating the feature matrix.
4. Run the script: Use the command line to run the script in either 'create' or 'search' mode.
##### Create Feature Matrix
To create a feature matrix from a dataset of images, run:
```
python similar_image_search.py --mode create --dataset /path/to/your/dataset --output /path/to/save/feature_matrix.pkl
```
Replace /path/to/your/dataset with the path to your dataset folder and /path/to/save/feature_matrix.pkl with the desired output file path.
##### Search Similar Images
To search for similar images using a query image, run:
```
python similar_image_search.py --mode search --output /path/to/save/feature_matrix.pkl --query /path/to/query/image.jpg
```
Replace /path/to/save/feature_matrix.pkl with the path to the saved feature matrix file and /path/to/query/image.jpg with the path to your query image.

### Example Commands
```sh
# Create feature matrix
python similar_image_search.py --mode create --dataset ./images --output ./feature_matrix.pkl

# Search similar images
python similar_image_search.py --mode search --output ./feature_matrix.pkl --query ./query_image.jpg
```

Make sure to adjust the paths according to your file structure.

## Solution Exmaple
<img src=https://github.com/jatiink/FlixStock-Assignment/assets/97089717/f03d8bab-360d-4494-8aaf-569f1769efa9
 alt="drawing" width="1000"/>

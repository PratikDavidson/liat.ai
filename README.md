# Player Re-Identification in Sports Footage

This repository provides a solution to one of the task assigned to assess the ability to solve real-world computer vision problem in sport analytics. 

## Task: Re-Identification in a Single Feed

Identification of each player ensuring same player retains the same ID even after going out of view in a given 15-second video.

### Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Install [**Miniconda**](https://www.anaconda.com/docs/getting-started/miniconda/install)

### Installation & Setup

_Below are steps to follow in setting up the project._

1. Create a new conda environment
```
conda create --name pytorch python=3.10
```
2. Activate the environment
```
conda activate pytorch
```
3. Install dependencies
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu[CUDA_VERSION] 
pip install ultralytics
pip install streamlit
pip install ipykernel
```
>**Note:** Replace [CUDA_VERSION] with the version of CUDA installed on your system. You can find this version by running the nvidia-smi command in your terminal, which displays the driver and CUDA version currently in use by your GPU.
Example: If CUDA Version is 12.8, then replace [CUDA_VERSION] with 128 - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
4. Directory Structure
```
liat.ai
|   .gitignore
|   app.py
|   main.py
|   README.md
|   requirements.txt
|   Soccer Player Re-Identification Report.pdf
|   test.ipynb
|
+---assets
|       demo.gif
|
+---data
|       15sec_input_720p.mp4
|
+---models
|       best.pt
|
\---runs
    \---detect
        \---track
                15sec_input_720p.avi
```
5. Use the following command to run main.py
```
python main.py
```
6. Use the following command to run streamlit app
```
streamlit run app.py
```

### Demo

![demo](./assets/demo.gif)
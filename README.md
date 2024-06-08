# 🌟 3dgs_render_python

English | [中文](assets/README_ch.md)

## 🚀 Introduction
**3dgs_render_python** is a project aimed at reimplementing the CUDA code part of [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) using Python. As a result, we have not only preserved the core functionality of the algorithm but also greatly enhanced the readability and maintainability of the code.

### 🌈 Advantages
- **Transparency**: Rewriting CUDA code in Python makes the internal logic of the algorithm clearer, facilitating understanding and learning.
- **Readability**: For beginners and researchers, this is an excellent opportunity to delve into parallel computing and 3DGS algorithms.

### 🔍 Disadvantages
- **Performance**: Since the project uses the CPU to simulate tasks originally handled by the GPU, the execution speed is slower than the native CUDA implementation.
- **Resource Consumption**: Simulating GPU operations with the CPU may lead to high CPU usage and memory consumption.

### 🛠️ Objective
The goal of this project is to provide an implementation of the 3DGS rendering part algorithm that is easier to understand and to offer a platform for users who wish to learn and experiment with 3D graphics algorithms without GPU hardware support.

## 📚 Applicable Scenarios
- **Education and Research**: Providing the academic community with the opportunity to delve into the study of 3DGS algorithms.
- **Personal Learning**: Helping individual learners understand the complexities of parallel computing and 3DGS.

Through **3dgs_render_python**, we hope to stimulate the community's interest in 3D graphics algorithms and promote broader learning and innovation.

## 🔧 Quick Start

### Installation Steps

```bash
# Clone the project using Git
git clone https://github.com/SY-007-Research/3dgs_render_python.git 

# Enter the project directory
cd 3dgs_render_python

# install requirements
pip install -r requirements.txt
```

### Running the Project

```bash
# Transformation demo
python transformation.py
```


|transformation 3d|transformation 2d|
|---|---|
|<img src="assets\transformation_3d.png" width = 300 height = 200>| <img src="assets\tranformation_2d.png" width = 200 height = 200>|

```bash
# 3DGS demo
python 3dgs.py
```

<img src="assets\3dgs.png" width = 300 height = 200>

## 🏅 Support

If you like this project, you can support us in the following ways:

- [GitHub Star](https://github.com/SY-007-Research/3dgs_render_python)
- [bilibili](https://space.bilibili.com/644569334)

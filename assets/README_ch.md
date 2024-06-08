# 🌟 3dgs_render_python

## 🚀 简介
**3dgs_render_python** 旨在将[3DGS](https://github.com/graphdeco-inria/gaussian-splatting)中的CUDA代码部分用Python重新实现。由此，我们不仅保留了算法的核心功能，还极大地提高了代码的可读性和可维护性。

### 🌈 优势
- **透明性**: 使用Python重写CUDA代码，使得算法的内部逻辑更加清晰，便于理解和学习。
- **易读性**: 对于初学者和研究者来说，这是一个深入理解并行计算和3dgs算法的绝佳机会。

### 🔍 缺点
- **性能**: 由于使用CPU来模拟原本由GPU处理的任务，项目在执行速度上不如原生CUDA实现，速度慢。
- **资源消耗**: CPU模拟GPU操作可能会导致较高的CPU使用率和内存消耗。

### 🛠️ 目标
本项目的目标是提供一个更加易于理解的3DGS的渲染部分算法实现，同时为那些希望在没有GPU硬件支持的情况下学习和实验3D图形算法的用户提供一个平台。


## 📚 适用场景
- 教育和研究：为学术界提供深入研究3DGS算法的机会。
- 个人学习：帮助个人学习者理解并行计算和3DGS的复杂性。

通过**3dgs_render_python**，我们希望能够激发社区对3D图形算法的兴趣，并促进更广泛的学习和创新。



## 🔧 快速开始



### 安装步骤

```bash
# 使用Git克隆项目
git clone https://github.com/SY-007-Research/3dgs_render_python.git

# 进入项目目录
cd 3dgs_render_python

# 安装依赖
pip install -r requirements.txt
```

### 运行项目

```bash
# transformation demo
python transformation.py
```


|transformation 3d|transformation 2d|
|---|---|
|<img src=".\transformation_3d.png" width = 300 height = 200>| <img src=".\tranformation_2d.png" width = 200 height = 200>|

```bash
# 3dgs demo
python 3dgs.py
```
<img src=".\3dgs.png" width = 300 height = 200>


## 🏅 支持

如果你喜欢这个项目，可以通过以下方式支持我们：

- [GitHub Star](https://github.com/SY-007-Research/3dgs_render_python)
- [bilibili](https://space.bilibili.com/644569334?spm_id_from=333.1296.0.0)
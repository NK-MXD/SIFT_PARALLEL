# 仓库说明

本仓库是郭坤昌，孟笑朵合作完成的计算机视觉课程设计作业——SIFT高性能化与特定场景应用。

## 目录说明

```
./ref 参考CUDA代码实现
./migrate-sift-opencv249 参考sift实现算法
./RSIR-SIFT 领域特定SIFT算法, 主要为SAR-SIFT算法实现
./sift-cuda CUDA版本sift算法实现
./sift CPU并行化sift算法实现
```
## 运行方式

前往对应目录, 按需求修改`CMakeLists.txt`文件, 例如运行CPU并行化sift算法:

```
$ cd sift/
$ mkdir build
$ cd build
$ cmake build .
$ make
$ ./sift
```

## CPU并行结果图像
<figure>
<img src="https://image-1305894911.cos.ap-beijing.myqcloud.com/Obsidian/202306071505377.png" width=200/>
<img src="https://image-1305894911.cos.ap-beijing.myqcloud.com/Obsidian/202306071505279.png" width=200/>
</figure>

<figure>
<img src="https://image-1305894911.cos.ap-beijing.myqcloud.com/Obsidian/202306071505762.png" width=200/>
<img src="https://image-1305894911.cos.ap-beijing.myqcloud.com/Obsidian/202306071505422.png" width=200/>
</figure>

## SAR-SIFT算法结果图像

![正确匹配特征点连线图.jpg|500](https://image-1305894911.cos.ap-beijing.myqcloud.com/Obsidian/202306071506831.jpg)

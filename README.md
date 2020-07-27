### Dependency

   1. opencv

   2. boost: sudo apt install libboost-all-dev

   3. eigen: sudo apt install libeigen3-dev

   4. pangolin: 

        sudo apt install libglew-dev

        git clone https://github.com/stevenlovegrove/Pangolin.git

        cd Pangolin

        mkdir build && cd build

        cmake ..

        make -j4

### newly added!!
    5. libnabo:
        cd omni_vslam/Thirdparty/libnabo
        mkdir build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo 
        make
        sudo make install
    6. 安装 nvidia显卡驱动   参考：https://blog.csdn.net/qq_32408773/article/details/84111244
         如果安装失败可能是ubuntu内核版本过高，一个参考是 5.3.0-62-generic ubuntu内核 + NVIDIA-Linux-x86_64-430.50.run (其他版本的显卡驱动也可以)
         1.Would you like to register the kernel module souces with DKMS? This will allow DKMS to automatically build a new module, if you install a different kernel later?  选择 No 继续。
         2.Nvidia's 32-bit compatibility libraries? 选择 No 继续。
         3.Would you like to run the nvidia-xconfigutility to automatically update your x configuration so that the NVIDIA x driver will be used when you restart x? Any pre-existing x confile will be backed up.  选择 Yes  继续
    7. 安装 cuda10.0 
            已下载（omni_vslam/Thirdparty/tensorflow_tools/cuda_10.0.130_410.48_linux.run）
            参考：https://blog.csdn.net/qq_32408773/article/details/84112166
            注意卸载之前的cuda版本
    8. 安装 cudnn-10.0v7.4.1.5 
            cd omni_vslam/Thirdparty/tensorflow_tools/cudnn-10.0-linux-x64-v7.4.1.5
            sudo cp cuda/include/cudnn.h /usr/local/cuda/include/ 
            sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/ 
            sudo chmod a+r /usr/local/cuda/include/cudnn.h 
    9.安装  protobuf-3.7.1 
            cd omni_vslam/Thirdparty/tensorflow_tools/protobuf-3.7.1
            ./autogen.sh
            ./configure
            make
            sudo make install

### 如果你使用的不是ubuntu18.04 或者 运行slam的时候出现tensorflow 链接错误，那么需要重新将tensorflow编译为.so文件，如下：
    10.安装 bazel  （此处提供的是x86_64版本的)  注意安装路径不能有中文！ 最后一行命令只在当前终端有效，建立把这行命令写入 ~/.bashrc 中
            cd omni_vslam/Thirdparty/tensorflow_tools
            sudo apt-get install openjdk-8-jdk
            chmod +x bazel-0.24.1-installer-linux-x86_64.sh
            ./bazel-0.24.1-installer-linux-x86_64.sh --user 
            export PATH="$PATH:$HOME/bin"     
    11.编译tensorflow
            cd omni_vslam/Thirdparty/tensorflow_r1.14
            ./configure   #进行配置，注意开启gpu模式，选项可以参考 https://blog.csdn.net/broliao/article/details/104545148
            bazel build --config=opt --config=cuda //tensorflow:libtensorflow_cc.so
            cd omni_vslam/Thirdparty/tensorflow_r1.14/bazel-bin/tensorflow   查看是否有libtensorflow_framework.so，没有的的话将其中最相似的改为这个文件名

            

      
    



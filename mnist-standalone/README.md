# tensorrt-cpp

Cuda-12.4, TensorRT 10.0.1.
Tested on RTX 3090

```
g++ mnist.cpp server.cpp -I/usr/lib/x86_64-linux-gnu -L /usr/lib/x86_64-linux-gnu `pkg-config --cflags --libs cuda-12.4` `pkg-config --cflags --libs cudart-12.4` -lnvinfer -lnvonnxparser -pthread
```
 g++ -std=c++11 -Wall -Wextra -Werror examples/liveness.cpp   $(pkg-config --cflags --libs opencv4)   -o liveness_example

 ./liveness_example model.onnx image.jpg \
  left_eye_x left_eye_y right_eye_x right_eye_y nose_x nose_y \
  left_mouth_x left_mouth_y right_mouth_x right_mouth_y



  ./liveness_example liveness.onnx  Sheldon05.jpg 645 268 772 282 692 354 647 420 735 428 
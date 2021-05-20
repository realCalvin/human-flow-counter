# human-flow-counter
---
## Object Movement Detection
```python human_counter.py```
- Difference in frames -> Grayscale -> Blur -> Dilate -> Contour
- Does not work well with clusters of humans--proceed with deep learning approach
---
## Deep Learning Detection (optimal approach)
```python deep_learning.py```
- Pass video through a pre-trained Caffe model via OpenCV DNN Module
- Obtain detections (coordinates) and confidence
- Utilize Euclidean Distance for unique human IDs to ensure each human is only counted once
---
Note: Ensure that you have [AWS CLI and SDK configured](https://docs.aws.amazon.com/rekognition/latest/dg/setup-awscli-sdk.html).

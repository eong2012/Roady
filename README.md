# Roady
![alt text][image1]
### Real time road segmentation for your smartphone!

[//]: # (Image References)
[image1]: ./padded_icon.png

### Getting started
Put your graph definition (`.pb`) in the `assets` directory and run the app. For more information on how to train and export the graph take a look at [this repository](https://github.com/see--/P12-Semantic-Segmentation). All of the app's logic is implemented in `FCNPredictActivity.java`. I put most of post-processing (e.g. argmax, resizing, alpha-blending) in the graph so you might have to change the `export_for_roady.py` script.
**Note**: I have never worked with Java. Thus, most of the code is copied from these two codelabs:
- [TensorFlow for Poets 2](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/index.html?index=..%2F..%2Findex#0)
- [Android & TensorFlow: Artistic Style Transfer](https://codelabs.developers.google.com/codelabs/tensorflow-style-transfer-android/index.html?index=..%2F..%2Findex#0)
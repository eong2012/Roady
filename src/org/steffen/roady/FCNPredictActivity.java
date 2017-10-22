package org.steffen.roady;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.Trace;
import android.util.Size;
import android.view.Display;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.steffen.roady.OverlayView.DrawCallback;
import org.steffen.roady.env.ImageUtils;
import org.steffen.roady.env.Logger;

/**
 * Created by See-- on 19.10.17.
 * Mostly copied from:
 * https://codelabs.developers.google.com/codelabs/tensorflow-style-transfer-android/index.html?index=..%2F..%2Findex#0
 * https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/index.html?index=..%2F..%2Findex#0
 */

public class FCNPredictActivity extends CameraActivity
  implements OnImageAvailableListener {

  private static final Logger LOGGER = new Logger();

  private TensorFlowInferenceInterface inferenceInterface;

  private static final String MODEL_FILE = "file:///android_asset/optimized.pb";

  private static final String INPUT_NODE = "rgb_preview_input";
  private static final String OUTPUT_NODE = "rgb_output_blended";


  private static final Size DESIRED_PREVIEW_SIZE = new Size(1280, 720);
  private static final int INPUT_HEIGHT = 720;
  private static  final int INPUT_WIDTH = 1280;
  private boolean initializedBuffers = false;
  private Integer sensorOrientation;
  private int previewWidth = 0;
  private int previewHeight = 0;
  // we get yuv -> we have to convert to rgb first
  private byte[][] yuvBytes;
  // converted bytes
  private int[] rgbBytes = null;
  private Bitmap rgbFrameBitmap = null;
  // input to the network
  private Bitmap croppedBitmap = null;

  private int[] intValues;
  private float[] floatValues;
  private Bitmap textureCopyBitmap;

  private boolean computing = false;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  @Override
  public void onCreate(final Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_stylize;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);
    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    final Display display = getWindowManager().getDefaultDisplay();
    final int screenOrientation = display.getRotation();

    LOGGER.d("Sensor orientation: %d, Screen orientation: %d", rotation, screenOrientation);

    sensorOrientation = rotation + screenOrientation;
    addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(Canvas canvas) {
            renderView(canvas);
          }
        }
    );
  }

  private void renderView(final Canvas canvas) {
    LOGGER.d("renderView");
    final Bitmap texture = textureCopyBitmap;

    if(texture != null) {
      final Matrix matrix = new Matrix();
      float scaleWidth = (float) canvas.getWidth() / texture.getWidth();
      float scaleHeight = (float) canvas.getHeight() / texture.getHeight();
      matrix.postScale(scaleWidth, scaleHeight);
      canvas.drawBitmap(texture, matrix, new Paint());  // matrix
    }
  }

  @Override
  public void onImageAvailable(final ImageReader reader) {
    Image image = null;

    try {
      image = reader.acquireLatestImage();

      if(image == null) {
        return;
      }
      if(computing) {
        image.close();
        return;
      }

      if(!initializedBuffers) {
        rgbBytes = new int[previewWidth * previewHeight];
        rgbFrameBitmap = Bitmap.createBitmap(
            previewWidth,
            previewHeight,
            Bitmap.Config.ARGB_8888);

        croppedBitmap = Bitmap.createBitmap(
            INPUT_WIDTH,
            INPUT_HEIGHT,
            Bitmap.Config.ARGB_8888);

        frameToCropTransform =
            ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight,
                INPUT_WIDTH, INPUT_HEIGHT,
                0, true);  // sensorOrientation

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        yuvBytes = new byte[3][];

        intValues = new int[INPUT_HEIGHT * INPUT_WIDTH];
        floatValues = new float[INPUT_HEIGHT * INPUT_WIDTH * 3];
        initializedBuffers = true;
      }

      computing = true;
      Trace.beginSection("imageAvailable");
      final Plane[] planes = image.getPlanes();
      fillBytes(planes, yuvBytes);

      final int yRowStride = planes[0].getRowStride();
      final int uvRowStride = planes[1].getRowStride();
      final int uvPixelStride = planes[1].getPixelStride();

      ImageUtils.convertYUV420ToARGB8888(
          yuvBytes[0],
          yuvBytes[1],
          yuvBytes[2],
          previewWidth,
          previewHeight,
          yRowStride,
          uvRowStride,
          uvPixelStride,
          rgbBytes);

      image.close();
    } catch (final Exception e) {
      if(image != null) {
        image.close();
      }
      LOGGER.e(e, "Exception!");
      Trace.endSection();
      return;
    }
    LOGGER.d("image converted to rgb!");


    rgbFrameBitmap.setPixels(
        rgbBytes, 0, previewWidth, 0, 0,
        previewWidth, previewHeight);

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            predict(croppedBitmap);
            textureCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            requestRender();
            computing = false;
          }
        });
    Trace.endSection();
  }

  private void predict(final Bitmap bitmap) {
    // put the bitmap pixels into intValues
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    LOGGER.d("Normalizing image!");
    // Note: Normalization is done in the graph
    // Just cast to float
    for(int i = 0; i < intValues.length; i++) {
      final int val = intValues[i];
      floatValues[i * 3 + 0] = (val >> 16) & 0xFF;
      floatValues[i * 3 + 1] = (val >> 8) & 0xFF;
      floatValues[i * 3 + 2] = val & 0xFF;
    }
    // feed tensor in [N, H, W, C] format
    inferenceInterface.feed(
        INPUT_NODE,
        floatValues,
        1,
        bitmap.getHeight(),
        bitmap.getWidth(),
        3);

    // sess.run() ...
    inferenceInterface.run(new String[] {OUTPUT_NODE}, isDebug());
    inferenceInterface.fetch(OUTPUT_NODE, floatValues);
    // argmax, blending etc. is all done inside the graph
    // Just cast to int
    for(int i = 0; i < intValues.length; i++) {
      intValues[i] =
          0xFF000000
              | (((int) (floatValues[i * 3])) << 16)
              | (((int) (floatValues[i * 3 + 1])) << 8)
              | ((int) (floatValues[i * 3 + 2]));
    }
    bitmap.setPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
  }
}

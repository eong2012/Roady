package org.tensorflow.demo;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Size;
import android.util.TypedValue;
import android.view.Display;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.demo.OverlayView.DrawCallback;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;

/**
 * Created by steffen on 19.10.17.
 */

public class FCNPredictActivity extends CameraActivity
  implements OnImageAvailableListener {

  private static final Logger LOGGER = new Logger();

  private TensorFlowInferenceInterface inferenceInterface;

  private static final String MODEL_FILE = "file:///android_asset/optimized.pb";

  private static final String INPUT_NODE = "rgb_preview_input";
  private static final String OUTPUT_NODE = "rgb_output_blended";

  private static final float TEXT_SIZE_DIP = 12;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(1280, 720);
  private static final int INPUT_HEIGHT = 720;
  private static  final int INPUT_WIDTH = 1280;
  private static  final int OUTPUT_HEIGHT = 720;
  private static final int OUTPUT_WIDTH = 1280;

  private Integer sensorOrientation;
  private int previewWidth = 0;
  private int previewHeight = 0;
  // we get yuv -> we have to convert to rgb first
  private byte[][] yuvBytes;
  // converted bytes
  private int[] rgbBytes = null;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap rgbFrameBitmapCopy = null;
  // input to the network
  private Bitmap croppedBitmap = null;

  private int[] intValues;
  private float[] floatValues;
  private int frameNum = 0;

  private Bitmap cropCopyBitmap;
  private Bitmap textureCopyBitmap;

  private boolean computing = false;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;
  private long lastProcessingTimeMs;

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
    LOGGER.d("Initializing inferenceINterface!");
    inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);
    LOGGER.d("Done - Initializing inferenceINterface!");

    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());

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
    // final Bitmap texture = rgbFrameBitmap;

    if(texture != null) {
      final Matrix matrix = new Matrix();
      /*final Matrix matrix = ImageUtils.getTransformationMatrix(
          texture.getWidth(), texture.getHeight(),
          previewWidth, previewHeight,
          0, true);  // sensorOrientation
      */
      float scaleFactor = Math.min(
          (float) canvas.getWidth() / texture.getWidth(),
          (float) canvas.getHeight() / texture.getHeight());
      float scaleWidth = (float) canvas.getWidth() / texture.getWidth();
      float scaleHeight = (float) canvas.getHeight() / texture.getHeight();
      matrix.postScale(scaleWidth, scaleHeight);

      LOGGER.d("Canvas: " + Integer.toString(canvas.getHeight()) + "x" + Integer.toString(canvas.getWidth()));
      LOGGER.d("texture: " + Integer.toString(texture.getHeight()) + "x" + Integer.toString(texture.getWidth()));

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

      // TODO(see--): we don't have to reset every time
      if(true) {
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
      }

      computing = true;
      Trace.beginSection("imageAvailable");
      final Plane[] planes = image.getPlanes();
      fillBytes(planes, yuvBytes);

      final int yRowStride = planes[0].getRowStride();
      final int uvRowStride = planes[1].getRowStride();
      final int uvPixelStride = planes[1].getPixelStride();
      LOGGER.d("before converting");

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

    // put the big rgb image (720 x 1280) in the small rgb image (160 x 576)
    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);  // croppedBitmap

            final long startTime = SystemClock.uptimeMillis();
            predict(croppedBitmap);  // croppedBitmap
            lastProcessingTimeMs = SystemClock.uptimeMillis();

            textureCopyBitmap = Bitmap.createBitmap(croppedBitmap);  // croppedBitmap

            requestRender();
            computing = false;
          }
        });
    Trace.endSection();
  }

  // we get an input image
  private void predict(final Bitmap bitmap) {
    ++frameNum;

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
        1,  // N
        bitmap.getHeight(),  // H
        bitmap.getWidth(),  // W
        3);  // C

    // sess.run ...
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

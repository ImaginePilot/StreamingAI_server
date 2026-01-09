package com.example.myapplication;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.Surface;
import android.view.TextureView;
import android.widget.Button;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;

import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class VideoStreamingActivity extends AppCompatActivity {

    private static final String TAG = "VideoStreamingActivity";
    private static final int REQUEST_PERMISSIONS = 200;
    private static final String VIDEO_SERVER_URL = "http://imaginepilot.xyz:5000";
    private static final String AUDIO_SERVER_URL = "http://imaginepilot.xyz:5001";
    
    // Video settings
    private static final int TARGET_FPS = 15;
    private static final int JPEG_QUALITY = 70;

    // Audio settings
    private static final int SAMPLE_RATE = 44100;
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;
    private static final int BUFFER_SIZE = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT);

    private TextureView textureView;
    private Button streamButton;
    private TextView logTextView;

    private CameraDevice cameraDevice;
    private CameraCaptureSession cameraCaptureSession;
    private CaptureRequest.Builder captureRequestBuilder;

    private HandlerThread backgroundThread;
    private Handler backgroundHandler;

    private volatile boolean isStreaming = false;
    private OkHttpClient httpClient;

    // Video sending mechanism
    private BlockingQueue<byte[]> frameQueue;
    private Thread videoSenderThread;
    private long lastFrameTime = 0;
    private final long frameInterval = 1000 / TARGET_FPS;

    // Audio recording mechanism
    private AudioRecord audioRecord;
    private Thread audioThread;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_video_streaming);

        httpClient = new OkHttpClient.Builder()
                .connectTimeout(5, TimeUnit.SECONDS)
                .writeTimeout(5, TimeUnit.SECONDS)
                .readTimeout(5, TimeUnit.SECONDS)
                .build();

        frameQueue = new ArrayBlockingQueue<>(2);

        textureView = findViewById(R.id.texture_view);
        streamButton = findViewById(R.id.stream_button);
        logTextView = findViewById(R.id.log_text_view);
        logTextView.setMovementMethod(new ScrollingMovementMethod());

        streamButton.setOnClickListener(v -> {
            if (isStreaming) {
                stopStreaming();
            } else {
                startStreaming();
            }
        });

        checkAndRequestPermissions();

        textureView.setSurfaceTextureListener(textureListener);
    }

    private void checkAndRequestPermissions() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED ||
            ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{
                    Manifest.permission.CAMERA,
                    Manifest.permission.RECORD_AUDIO
            }, REQUEST_PERMISSIONS);
        }
    }

    private void logMessage(final String message) {
        runOnUiThread(() -> {
            logTextView.append(message + "\n");
            // Auto scroll to bottom
            final int scrollAmount = logTextView.getLayout().getLineTop(logTextView.getLineCount()) - logTextView.getHeight();
            if (scrollAmount > 0)
                logTextView.scrollTo(0, scrollAmount);
            else
                logTextView.scrollTo(0, 0);
        });
    }

    private void startStreaming() {
        logMessage("Attempting to start stream...");
        new Thread(() -> {
            try {
                // Start Video Session
                Request videoStartReq = new Request.Builder()
                        .url(VIDEO_SERVER_URL + "/stream/start")
                        .post(RequestBody.create(new byte[0]))
                        .build();
                
                // Start Audio Session
                JSONObject audioConfig = new JSONObject();
                try {
                    audioConfig.put("sample_rate", SAMPLE_RATE);
                    audioConfig.put("channels", 1);
                    audioConfig.put("sample_width", 2);
                } catch (JSONException e) {
                    e.printStackTrace();
                }

                Request audioStartReq = new Request.Builder()
                        .url(AUDIO_SERVER_URL + "/stream/start")
                        .post(RequestBody.create(audioConfig.toString(), MediaType.parse("application/json")))
                        .build();

                try (Response vResponse = httpClient.newCall(videoStartReq).execute();
                     Response aResponse = httpClient.newCall(audioStartReq).execute()) {
                    
                    if (vResponse.isSuccessful() && aResponse.isSuccessful()) {
                        runOnUiThread(() -> {
                            isStreaming = true;
                            streamButton.setText("Stop Streaming");
                            logMessage("Video and Audio streaming started.");
                        });
                        
                        // Brief delay to ensure session is ready on server
                        try {
                            Thread.sleep(100);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }
                        
                        startVideoSenderThread();
                        startAudioRecording();
                    } else {
                        logMessage("Failed to start streaming. Video: " + vResponse.code() + ", Audio: " + aResponse.code());
                    }
                }
            } catch (IOException e) {
                logMessage("Failed to start streaming: " + e.getMessage());
                Log.e(TAG, "Failed to start streaming", e);
            }
        }).start();
    }

    private void stopStreaming() {
        logMessage("Stopping stream...");
        isStreaming = false;

        // Stop video sender thread
        if (videoSenderThread != null) {
            videoSenderThread.interrupt();
            videoSenderThread = null;
        }
        frameQueue.clear();

        // Stop audio thread
        if (audioThread != null) {
            audioThread.interrupt();
            audioThread = null;
        }
        if (audioRecord != null) {
            audioRecord.stop();
            audioRecord.release();
            audioRecord = null;
        }

        new Thread(() -> {
            try {
                Request videoStopReq = new Request.Builder()
                        .url(VIDEO_SERVER_URL + "/stream/stop")
                        .post(RequestBody.create(new byte[0]))
                        .build();
                Request audioStopReq = new Request.Builder()
                        .url(AUDIO_SERVER_URL + "/stream/stop")
                        .post(RequestBody.create(new byte[0]))
                        .build();
                
                httpClient.newCall(videoStopReq).execute();
                httpClient.newCall(audioStopReq).execute();
                logMessage("Streaming stopped.");
            } catch (IOException e) {
                logMessage("Failed to stop streaming: " + e.getMessage());
                Log.e(TAG, "Failed to stop streaming", e);
            }
        }).start();

        streamButton.setText("Start Streaming");
    }

    private void startVideoSenderThread() {
        videoSenderThread = new Thread(() -> {
            while (isStreaming && !Thread.currentThread().isInterrupted()) {
                try {
                    byte[] frameData = frameQueue.poll(100, TimeUnit.MILLISECONDS);
                    if (frameData != null) {
                        sendFrameData(frameData);
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        });
        videoSenderThread.start();
    }

    private void sendFrameData(byte[] imageData) {
        if (!isStreaming) return;  // Don't send if session stopped
        
        try {
            RequestBody requestBody = RequestBody.create(imageData, MediaType.parse("image/jpeg"));
            Request request = new Request.Builder()
                    .url(VIDEO_SERVER_URL + "/camera/frame")
                    .post(requestBody)
                    .build();

            try (Response response = httpClient.newCall(request).execute()) {
                if (!response.isSuccessful()) {
                    Log.w(TAG, "Failed to send frame: " + response.code());
                }
            }
        } catch (IOException e) {
            Log.e(TAG, "Failed to send frame", e);
        }
    }

    private void startAudioRecording() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            return;
        }
        audioRecord = new AudioRecord(MediaRecorder.AudioSource.MIC, SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT, BUFFER_SIZE);
        audioRecord.startRecording();

        audioThread = new Thread(() -> {
            byte[] buffer = new byte[BUFFER_SIZE];
            while (isStreaming && !Thread.currentThread().isInterrupted()) {
                int read = audioRecord.read(buffer, 0, buffer.length);
                if (read > 0) {
                    byte[] audioData = new byte[read];
                    System.arraycopy(buffer, 0, audioData, 0, read);
                    sendAudioData(audioData);
                }
            }
        });
        audioThread.start();
    }

    private void sendAudioData(byte[] audioData) {
        if (!isStreaming) return;  // Don't send if session stopped
        
        try {
            RequestBody requestBody = RequestBody.create(audioData, MediaType.parse("application/octet-stream"));
            Request request = new Request.Builder()
                    .url(AUDIO_SERVER_URL + "/audio/chunk")
                    .post(requestBody)
                    .build();

            try (Response response = httpClient.newCall(request).execute()) {
                if (!response.isSuccessful()) {
                    Log.w(TAG, "Failed to send audio chunk: " + response.code());
                }
            }
        } catch (IOException e) {
            Log.e(TAG, "Failed to send audio chunk", e);
        }
    }

    private final TextureView.SurfaceTextureListener textureListener = new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(@NonNull SurfaceTexture surface, int width, int height) {
            openCamera();
        }

        @Override
        public void onSurfaceTextureSizeChanged(@NonNull SurfaceTexture surface, int width, int height) {}

        @Override
        public boolean onSurfaceTextureDestroyed(@NonNull SurfaceTexture surface) {
            return false;
        }

        @Override
        public void onSurfaceTextureUpdated(@NonNull SurfaceTexture surface) {
            if (isStreaming) {
                long currentTime = System.currentTimeMillis();
                if (currentTime - lastFrameTime >= frameInterval) {
                    lastFrameTime = currentTime;
                    captureAndQueueFrame();
                }
            }
        }
    };

    private void captureAndQueueFrame() {
        Bitmap bitmap = textureView.getBitmap();
        if (bitmap != null) {
            try {
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                bitmap.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, baos);
                byte[] imageData = baos.toByteArray();

                if (!frameQueue.offer(imageData)) {
                    frameQueue.poll();
                    frameQueue.offer(imageData);
                }
            } finally {
                bitmap.recycle();
            }
        }
    }

    private void openCamera() {
        CameraManager manager = (CameraManager) getSystemService(CAMERA_SERVICE);
        try {
            String cameraId = manager.getCameraIdList()[0];
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                return;
            }
            manager.openCamera(cameraId, stateCallback, null);
        } catch (CameraAccessException e) {
            logMessage("Failed to open camera: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private final CameraDevice.StateCallback stateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(@NonNull CameraDevice camera) {
            cameraDevice = camera;
            createCameraPreview();
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice camera) {
            cameraDevice.close();
        }

        @Override
        public void onError(@NonNull CameraDevice camera, int error) {
            if (cameraDevice != null) {
                cameraDevice.close();
                cameraDevice = null;
            }
        }
    };

    private void createCameraPreview() {
        try {
            SurfaceTexture texture = textureView.getSurfaceTexture();
            assert texture != null;
            Surface surface = new Surface(texture);
            captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            captureRequestBuilder.addTarget(surface);

            cameraDevice.createCaptureSession(Collections.singletonList(surface), new CameraCaptureSession.StateCallback() {
                @Override
                public void onConfigured(@NonNull CameraCaptureSession session) {
                    if (cameraDevice == null) {
                        return;
                    }
                    cameraCaptureSession = session;
                    try {
                        captureRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
                        cameraCaptureSession.setRepeatingRequest(captureRequestBuilder.build(), null, backgroundHandler);
                    } catch (CameraAccessException e) {
                        logMessage("Failed to create camera preview: " + e.getMessage());
                        e.printStackTrace();
                    }
                }

                @Override
                public void onConfigureFailed(@NonNull CameraCaptureSession session) {
                    logMessage("Camera configuration failed.");
                }
            }, null);
        } catch (CameraAccessException e) {
            logMessage("Failed to create camera preview: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private void startBackgroundThread() {
        backgroundThread = new HandlerThread("Camera Background");
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());
    }

    private void stopBackgroundThread() {
        if (backgroundThread != null) {
            backgroundThread.quitSafely();
            try {
                backgroundThread.join();
                backgroundThread = null;
                backgroundHandler = null;
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        startBackgroundThread();
        if (textureView.isAvailable()) {
            openCamera();
        } else {
            textureView.setSurfaceTextureListener(textureListener);
        }
    }

    @Override
    protected void onPause() {
        if (isStreaming) {
            stopStreaming();
        }
        closeCamera();
        stopBackgroundThread();
        super.onPause();
    }

    private void closeCamera() {
        if (cameraCaptureSession != null) {
            cameraCaptureSession.close();
            cameraCaptureSession = null;
        }
        if (cameraDevice != null) {
            cameraDevice.close();
            cameraDevice = null;
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_PERMISSIONS) {
            for (int result : grantResults) {
                if (result == PackageManager.PERMISSION_DENIED) {
                    logMessage("Required permissions denied.");
                    finish();
                    return;
                }
            }
        }
    }
}
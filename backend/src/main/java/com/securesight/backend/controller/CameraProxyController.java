package com.securesight.backend.controller;

import java.io.InputStream;
import java.net.URI;

import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.servlet.mvc.method.annotation.StreamingResponseBody;

@Controller
public class CameraProxyController {

    private static final String STREAM_URL = "http://192.168.0.109:5000/video";

    @GetMapping(value = "/api/camera-stream", produces = "multipart/x-mixed-replace;boundary=frame")
    public ResponseEntity<StreamingResponseBody> proxyCameraStream() {
        StreamingResponseBody stream = outputStream -> {
            try (InputStream inputStream = URI.create(STREAM_URL).toURL().openStream()) {
                byte[] buffer = new byte[4096];
                int bytesRead;
                while ((bytesRead = inputStream.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, bytesRead);
                    outputStream.flush();
                }
            } catch (Exception e) {
                System.err.println("[Camera Proxy] Error: " + e.getMessage());
            }
        };
        return ResponseEntity
        .ok()
        .header("Content-Type", "multipart/x-mixed-replace;boundary=frame")
        .body(stream);
    }
}
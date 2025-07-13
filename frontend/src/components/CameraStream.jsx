import React from 'react';

const CameraStream = () => {
  return (
    <div className="p-4">
      <h2 className="text-xl font-semibold mb-4">Live Object Detection Stream</h2>
      <div className="border rounded shadow-md overflow-hidden w-fit">
        <img
          src="http://localhost:8080/api/camera-stream"
          alt="Live Stream"
          className="max-w-full h-auto"
        />
      </div>
    </div>
  );
};

export default CameraStream;

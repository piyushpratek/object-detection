"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import { load as cocoSSDLoad, ObjectDetection as CocoSSDObjectDetection } from "@tensorflow-models/coco-ssd";
import * as tf from "@tensorflow/tfjs";
import { renderPredictions } from "@/utils/render-predictions";

type WebcamElement = Webcam & {
  video?: HTMLVideoElement;
};

type CanvasElement = HTMLCanvasElement & {
  width?: number;
  height?: number;
};

let detectInterval;

const ObjectDetection = () => {
  const [isLoading, setIsLoading] = useState(true);

  const webcamRef = useRef<WebcamElement>(null);
  const canvasRef = useRef<CanvasElement>(null);

  const runCoco = useCallback(async () => {
    setIsLoading(true); // Set loading state to true when model loading starts
    await tf.setBackend("webgl"); // Ensure TensorFlow.js uses the WebGL backend
    await tf.ready(); // Wait for the backend to be ready

    const net: CocoSSDObjectDetection = await cocoSSDLoad();
    setIsLoading(false); // Set loading state to false when model loading completes

    detectInterval = setInterval(() => {
      runObjectDetection(net); // will build this next
    }, 10);
  }, []);

  async function runObjectDetection(net: CocoSSDObjectDetection) {
    if (
      canvasRef.current &&
      webcamRef.current !== null &&
      webcamRef.current.video?.readyState === 4
    ) {
      canvasRef.current.width = webcamRef.current.video.videoWidth;
      canvasRef.current.height = webcamRef.current.video.videoHeight;

      // find detected objects
      const detectedObjects = await net.detect(
        webcamRef.current.video,
        undefined,
        0.6
      );

      // console.log(detectedObjects);

      const context = canvasRef.current.getContext("2d");
      if (context) {
        renderPredictions(detectedObjects, context);
      }
    }
  }

  const showmyVideo = () => {
    if (
      webcamRef.current !== null &&
      webcamRef.current.video?.readyState === 4
    ) {
      const myVideoWidth = webcamRef.current.video.videoWidth;
      const myVideoHeight = webcamRef.current.video.videoHeight;

      webcamRef.current.video.width = myVideoWidth;
      webcamRef.current.video.height = myVideoHeight;
    }
  };

  useEffect(() => {
    runCoco();
    showmyVideo();
  }, [runCoco]);

  return (
    <div className="mt-8">
      {isLoading ? (
        <div className="gradient-text">Loading AI Model...</div>
      ) : (
        <div className="relative flex justify-center items-center gradient p-1.5 rounded-md">
          {/* webcam */}
          <Webcam
            ref={webcamRef}
            className="rounded-md w-full lg:h-[720px]"
            muted
          />
          {/* canvas */}
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 z-99999 w-full lg:h-[720px]"
          />
        </div>
      )}
    </div>
  );
};

export default ObjectDetection;

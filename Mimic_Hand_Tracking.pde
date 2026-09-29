// The amount of things I imported is impressive already

import processing.serial.*;
import gab.opencv.*;
import processing.video.*;
import java.awt.*;
import java.util.List;
import java.util.ArrayList;

Serial myPort;

int pointerState = 0, middleState = 0, indexState = 0, pinkyState = 0;

/* 

0 = down, 1 = up --> this was found empirically through values I did with another processing program
                     where I would send values in and then see how it corresponded to the mechanical hand

*/

int stableFingerCount = 0; // Stores the stable number of detected fingers
int lastStableFingerCount = 0; // Last stable finger count
int stableTime = 1500; // x number of seconds for stable detection
int lastChangeTime = 0; // Time when the stable finger count was last detected

// camera stuff you need using open CV (a good amount of YouTube videos on it definitely helped)

OpenCV opencv;
Capture cam;
PImage img;
int fingerCount = 0; // Fingers detected
ArrayList<PVector> dots = new ArrayList<PVector>(); // To store the positions of the dots

void setup()
{
  
  size(640, 480);
  String[] cameras = Capture.list();
  
  if (cameras.length == 0)
  {
    println("No camera found");    // Regular camera check
    exit();
  }
  
  // Turns the camera on after finding it
  cam = new Capture(this, cameras[0]);
  cam.start();
  img = createImage(640, 480, RGB);
  opencv = new OpenCV(this, 640, 480);
  
  myPort = new Serial(this, "COM6", 9600);  // Regular com port

}

void draw() {
    if (cam.available())
    {
        cam.read();
        opencv.loadImage(cam);
        opencv.gray();
        opencv.threshold(110);  // Adjust for contrast tuning

        img.copy(opencv.getSnapshot(), 0, 0, cam.width, cam.height, 0, 0, img.width, img.height);
        img.updatePixels();

        // Flip the camera image
        pushMatrix();
        translate(width, 0);
        scale(-1, 1);
        image(img, 0, 0);
        popMatrix();

        // Find contours
        opencv.findContours();
        fingerCount = fingertips(opencv);

        // Draw hand outline
        List<Contour> contours = opencv.findContours();
        Contour largestContour = null;
        float maxArea = 5000;

        for (Contour contour : contours) {
            if (contour.area() > maxArea) {
                maxArea = contour.area();
                largestContour = contour;
            }
        }

        // If a hand is detected, outline it
        if (largestContour != null) {
            stroke(0, 255, 255); // Light blue outline
            strokeWeight(3);
            noFill();

            // Mirror the contour **DOWNWARD** (across Y-axis)
            pushMatrix();
            translate(width, 0);  // Shift to the right for X-mirroring
            scale(-1, 1);         // Flip horizontally to match expected reflection
            largestContour.draw();
            popMatrix();
        }
    }

    // Display the detected finger count
    fill(25, 105, 200);
    textSize(20);
    text("Fingers Detected: " + fingerCount, 20, 20);

    // Debounce-like stabilization for consistent finger detection
    if (fingerCount == stableFingerCount) {
        if (millis() - lastChangeTime > stableTime) {
            if (fingerCount != lastStableFingerCount) {
                updateFinger(fingerCount);
                lastStableFingerCount = fingerCount;
                lastChangeTime = millis();
            }
        }
    } else {
        stableFingerCount = fingerCount;
        lastChangeTime = millis();
    }
}




// Very complicated function, thank the internet for opensource projects for vision projects. Had to modify an original facial software recognition program to do this
int fingertips(OpenCV opencv)
{
    List<Contour> contours = opencv.findContours();
    Contour largestContour = null;
    float maxArea = 5000;
  
    // Find the largest contour based on area
    for (Contour contour : contours) {
        if (contour.area() > maxArea) {
            maxArea = contour.area();
            largestContour = contour;
        }
    }

    // Store detected fingertip positions
    ArrayList<PVector> currentDots = new ArrayList<PVector>();

    if (largestContour != null) {
        Contour hullContour = largestContour.getConvexHull();
        List<PVector> hullPoints = hullContour.getPoints();

        for (PVector point : hullPoints) {
            point.x = width - point.x;  // Mirror image horizontally
        }

        for (PVector b : hullPoints) {
            if (b.y < height / 1.5) {  // Filter out lower points
                currentDots.add(b);
            }
        }
    }

    // Group fingertips that are close together (distance < 50px)
    ArrayList<PVector> clusterCenters = new ArrayList<>();

    while (!currentDots.isEmpty()) {
        PVector start = currentDots.remove(0);
        ArrayList<PVector> cluster = new ArrayList<>();
        cluster.add(start);

        for (int i = currentDots.size() - 1; i >= 0; i--) {
            if (PVector.dist(start, currentDots.get(i)) < 50) {
                cluster.add(currentDots.remove(i));
            }
        }

        // Compute average position of the cluster
        float avgX = 0, avgY = 0;
        for (PVector p : cluster) {
            avgX += p.x;
            avgY += p.y;
        }
        avgX /= cluster.size();
        avgY /= cluster.size();

        clusterCenters.add(new PVector(avgX, avgY));
    }

    // Draw bounding boxes for each detected fingertip
    for (PVector center : clusterCenters) {
        fill(0, 0, 255, 100);  // Semi-transparent blue
        stroke(0, 0, 255);     // Blue outline
        strokeWeight(2);
        rectMode(CENTER);
        rect(center.x, center.y, 50, 50);  // 50x50 bounding box
    }

    return clusterCenters.size();  // Number of detected fingertips
}



// Function to update the finger states based on the detected number of fingers (I know there is a more efficient way to do this other than 4 if statements but it works)
void updateFinger(int detectedFingers)
{
  if (detectedFingers == 4)
  {
    pointerState = 1;
    middleState = 1;
    indexState = 1;
    pinkyState = 1;
  }
  else if
  (detectedFingers == 3)
  {
    pointerState = 1;
    middleState = 1;
    indexState = 1;
    pinkyState = 0;
  }
  else if
  (detectedFingers == 2)
  {
    pointerState = 1;
    middleState = 1;
    indexState = 0;
    pinkyState = 0;
  }
  else if
  (detectedFingers == 1)
  {
    pointerState = 1;
    middleState = 0;
    indexState = 0;
    pinkyState = 0;
  }
  else
  {
    pointerState = 0;
    middleState = 0;
    indexState = 0;
    pinkyState = 0;
  }
  
  
  sendAngles();
}

void sendAngles() {
  // Send the updated states to the Arduino for actual motion --> Had to make function as I was having issues having it be a part of "updateFinger"
  
  int pointerAngle = pointerState == 1 ? 180 : 90; // These are the "up down" or "on off" states for each
  int middleAngle = middleState == 1 ? 180 : 90;
  int indexAngle = indexState == 1 ? 0 : 180;
  int pinkyAngle = pinkyState == 1 ? 120 : 0; // Slightly tuned pinky due to torque issues

  String angleData = pointerAngle + " " + middleAngle + " " + indexAngle + " " + pinkyAngle + "\n";
  myPort.write(angleData);  // Send to Arduino
}

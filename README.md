# YOLO Mini — Object Detection, Tracking & Movement Analysis

This repository represents my hands-on learning and implementation of a basic computer-vision pipeline using **YOLO11**, OpenCV, and Python.

The project started with object detection and gradually moved toward understanding what happens **after an object is detected**. Instead of treating detection as the final result, I explored how detected objects can be extracted, tracked across frames, analyzed for movement, and compared with other objects based on their spatial relationships.

The main idea behind this project is:

**Detection → Coordinates → Tracking → Movement → Interaction**

---

## Project Overview

Object detection answers a relatively simple question:

> **"What objects are present in this frame, and where are they?"**

A detector such as YOLO provides information such as the object's class, confidence score, and bounding box.

However, a single frame does not tell us whether an object is moving, whether it is the same object that appeared in the previous frame, or whether two objects are interacting.

This project explores those additional steps.

The YOLO model is used as the foundation for detecting objects. The detected bounding boxes are then converted into useful coordinate information. Those coordinates can be compared between consecutive frames to understand object movement and maintain a basic form of tracking.

The project also explores spatial relationships between bounding boxes using overlapping boundaries. This provides a simple way of reasoning about whether detected objects are close to or interacting with each other.

---

## Object Detection

The first major concept explored in this project is **object detection**.

Object detection is different from image classification.

Classification answers:

> "What is in this image?"

Object detection answers:

> "What objects are in this image, and where are they?"

YOLO provides a bounding box around each detected object:

```text
(x1, y1) ---------------- (x2, y1)
   |                          |
   |          Object          |
   |                          |
(x1, y2) ---------------- (x2, y2)
```

The coordinates describe the position of the object inside the image.

A detection also contains information such as:

- Object class
- Confidence score
- Bounding-box coordinates

These values form the foundation for the rest of the pipeline.

The `detection.py` and `extract_detection.py` files work around this stage, taking the output of the detection model and extracting information that can be used by other parts of the system.

---

## Working With Bounding-Box Coordinates

A bounding box becomes much more useful once its geometry is understood.

A detection can be represented using:

```text
x1, y1, x2, y2
```

where:

- `x1, y1` represent one corner of the box
- `x2, y2` represent the opposite corner

From these coordinates, the center of the detected object can be calculated:

```text
center_x = (x1 + x2) / 2
center_y = (y1 + y2) / 2
```

This gives the object a representative position:

```text
Object → (center_x, center_y)
```

For example:

```text
Frame 1 → Object center = (100, 200)
Frame 2 → Object center = (108, 205)
```

The actual image contains much more information, but the center coordinate provides a simple way to reason about the object's position.

The `track_coordinates.py` component focuses on working with these positions.

---

## From Detection to Tracking

Detection and tracking are two different concepts.

A detector can independently detect objects in every frame:

```text
Frame 1 → Detect
Frame 2 → Detect
Frame 3 → Detect
```

But simply detecting an object in each frame does not automatically mean that the program knows which detection belongs to which object over time.

Tracking introduces the idea of maintaining an object's identity or position across multiple frames.

For example:

```text
Frame 1 → Object A → (100, 200)
Frame 2 → Object A → (105, 203)
Frame 3 → Object A → (111, 207)
```

By comparing these positions, the program can reason that the object has continued moving through the scene.

The `tracking.py` component represents this transition from working with individual detections to working with objects across time.

This is an important step because the problem changes from:

> **"What is in this frame?"**

to:

> **"What is happening to this object over time?"**

---

## Movement Detection

Once object positions are available for consecutive frames, movement can be estimated by comparing their coordinates.

Conceptually:

```text
Previous position
        ↓
    (x1, y1)

        compared with

Current position
        ↓
    (x2, y2)

        ↓

Has the object moved enough?
```

For example:

```text
Previous → (100, 200)
Current  → (120, 210)
```

The position has changed, so the object can be considered moving.

However, simply checking whether the coordinates changed is not enough.

Object detection is not perfectly identical from frame to frame. Even a stationary object may receive slightly different bounding-box coordinates because of detector variation, camera movement, lighting, or other changes in the image.

Therefore, movement is better understood using a **threshold**.

Conceptually:

```text
Small coordinate change
        ↓
   Stationary

Large coordinate change
        ↓
     Moving
```

The `detect_movement.py` and `detection.py` components explore this idea of using changes in object position to classify objects as moving or stationary.

An important understanding from this implementation is that **YOLO detects the object, while movement is inferred from changes in the detection across time**.

---

## Stationary and Moving Objects

The movement analysis can be represented as:

```text
                 Object
                    |
          Compare positions
                    |
          +---------+---------+
          |                   |
   Small movement       Large movement
          |                   |
     Stationary             Moving
```

This makes it possible to add a layer of interpretation on top of the raw object detection.

Instead of only producing:

```text
Person detected
```

the system can reason about:

```text
Person detected → currently moving
```

or:

```text
Person detected → currently stationary
```

This is one of the main transitions in the project from **object detection** toward **video analysis**.

---

## Object Interaction

The project also explores relationships between multiple detected objects.

If two objects are detected in the same frame, their bounding boxes can be compared.

For example:

```text
+-------------------+
|      Object A     |
|          +--------+---------+
|          |        | Object B|
|          |        |         |
+----------+--------+---------+
```

If their bounding boxes overlap, that overlap can be used as a simple indication that the objects are spatially close or potentially interacting.

The `object_interaction.py` component explores this idea using overlapping bounding boundaries.

The basic concept is:

```text
Object A bounding box
          +
Object B bounding box
          ↓
Compare their boundaries
          ↓
Do the regions overlap?
```

This does not automatically mean that two objects are physically interacting in the real world.

It only establishes a **spatial relationship in the image**.

For example, two people standing close together may have overlapping or nearby bounding boxes, but this alone does not prove that they are touching or interacting.

This distinction is important when moving from simple detection toward more meaningful scene understanding.

---

## The Complete Pipeline

The different concepts explored in this repository can be understood as one continuous pipeline:

```text
                 Video / Image
                       |
                       ↓
                YOLO Object Detection
                       |
                       ↓
             Bounding Box Information
                       |
                       ↓
               Extract Coordinates
                       |
                       ↓
              Track Across Frames
                       |
                       ↓
             Compare Object Positions
                       |
                       ↓
             Movement Classification
                       |
                       ↓
              Object Relationships
                       |
                       ↓
              Basic Scene Analysis
```

Each stage builds on information produced by the previous stage.

The detector provides the objects.

The coordinates provide their positions.

Tracking provides information across time.

Movement analysis interprets changes in those positions.

Object interaction uses the spatial relationship between multiple detections.

---

## Repository Structure

```text
yolo-mini/
│
├── har_mini/
│   ├── detect_movement.py
│   ├── detection.py
│   ├── extract_detection.py
│   ├── object_interaction.py
│   ├── track_coordinates.py
│   ├── tracking.py
│   └── yolo11n.pt
│
├── learning/
│
├── src/
│
├── .gitignore
├── .python-version
├── README.md
├── pyproject.toml
├── uv.lock
└── yolo11n.pt
```

The `har_mini` directory contains the main computer-vision work.

The Python files are separated according to the different stages of processing, while the YOLO model weights provide the trained detection model used by the project.

---

## What This Project Represents

This project is less about building a complicated production-ready tracking system and more about understanding how the pieces of a computer-vision system fit together.

The important progression is:

```text
Raw Image
   ↓
Detect Objects
   ↓
Understand Bounding Boxes
   ↓
Extract Coordinates
   ↓
Observe Objects Across Frames
   ↓
Detect Movement
   ↓
Compare Multiple Objects
   ↓
Understand Basic Spatial Interaction
```

The key idea is that **object detection is only the beginning**.

Once an object has been detected, its bounding box becomes a source of information that can be used to reason about position, movement, and relationships with other objects.

This repository represents my practical exploration of that process, starting from YOLO-based detection and extending it into basic tracking, movement analysis, and object interaction.

---

## Current Scope

The implementation is intentionally simple and focuses on understanding the underlying concepts rather than hiding everything behind a high-level tracking framework.

The project therefore provides a foundation for understanding how a video-analysis system can be constructed from basic components:

```text
Detection
    ↓
Geometry
    ↓
Temporal Comparison
    ↓
Movement
    ↓
Spatial Relationships
```

Understanding these fundamentals makes it easier to reason about more advanced computer-vision systems later, because higher-level systems are ultimately built around the same core ideas:

**detecting objects, representing them spatially, following them through time, and interpreting their relationships.**

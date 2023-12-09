# Fingerprint 2D-Pose Labeler

This project is a tool for annotating 2D poses in fingerprint images. It provides two modes for users:

- Direct Annotation: Users can directly mark the center and direction of fingerprints.

- Annotation by Manual Alignment: Users can obtain the pose by manually aligning two fingerprints, where the pose of one of them is known.

## File Format

The format of `pose2d` file is as follows: `center_x, center_y, direction_in_degree`.
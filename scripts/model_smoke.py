"""Load verified model and infer on blank pixels; never open a camera."""
def main():
    import numpy as np
    import mediapipe as mp
    from apps.desktop.camera import verified_model, create_detector
    with create_detector(verified_model()) as detector:
        result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
            data=np.zeros((480, 640, 3), dtype=np.uint8)), 1)
        assert len(result.face_landmarks) == 0
    print('Verified model loaded; blank-image inference passed; camera was NOT opened.')


if __name__ == '__main__':
    main()

import argparse
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
from platerec import Platerec
import os


def cv2_to_pil(cv2_img):
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def confidence_to_color(confidence):
    r = int(255 * (1 - confidence))
    g = int(255 * confidence)
    return (r, g, 0)


def annotate_images(image, output, font_size):
    boxes = output["boxes"]
    confidences = output["confidences"]
    words = output["words"]
    words_confidences = output["words_confidences"]

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        confidence = confidences[i]
        word_confidence = words_confidences[i]
        word = words[i]

        box_color = confidence_to_color(confidence)
        text_color = confidence_to_color(word_confidence)

        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
        draw.text((x1, y1 - font_size), f"{confidence:.2f}", fill=box_color, font=font)
        draw.text((x1, y2), word, fill=text_color, font=font)
        draw.text(
            (x1, y2 + font_size), f"{word_confidence:.2f}", fill=text_color, font=font
        )

    return image


def main():
    parser = argparse.ArgumentParser(
        description="Annotate video frames with detected plates."
    )
    parser.add_argument("video_path", help="Path to the video file.")
    parser.add_argument(
        "--font_size", type=int, default=20, help="Font size for annotations."
    )
    parser.add_argument(
        "--save_output", action="store_true", help="Save annotated video output."
    )
    args = parser.parse_args()

    platerec = Platerec(providers=["CUDAExecutionProvider"])
    cap = cv2.VideoCapture(args.video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.save_output:
        base_name, ext = os.path.splitext(args.video_path)
        output_video_path = f"{base_name}_platerec{ext}"
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    else:
        out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pil_image = cv2_to_pil(frame)
        output = platerec.detect_read(pil_image)
        if len(output["boxes"]) == 0:
            annotated_image = pil_image
        else:
            annotated_image = annotate_images(pil_image, output, args.font_size)
        frame = pil_to_cv2(annotated_image)

        if args.save_output:
            out.write(frame)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if args.save_output:
        out.release()
        print(f"Output video saved as: {output_video_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

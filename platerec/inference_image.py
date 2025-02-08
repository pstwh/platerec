import argparse

from PIL import Image, ImageDraw, ImageFont

from platerec import Platerec


def confidence_to_color(confidence):
    r = int(255 * (1 - confidence))
    g = int(255 * confidence)
    return (r, g, 0)


def annotate_images(image, output):
    boxes = output["boxes"]
    confidences = output["confidences"]
    words = output["words"]
    words_confidences = output["words_confidences"]

    draw = ImageDraw.Draw(image)
    font_size = 20
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
        draw.text(
            (x1, y1 - font_size),
            f"{confidence:.2f}",
            fill=box_color,
            font=font,
        )
        draw.text((x1, y2), word, fill=text_color, font=font)
        draw.text(
            (x1, y2 + font_size),
            f"{word_confidence:.2f}",
            fill=text_color,
            font=font,
        )

    return image


def main():
    parser = argparse.ArgumentParser(
        description="Annotate images with detected text and confidence scores."
    )
    parser.add_argument("input_image", type=str, help="Path to the input image file.")

    args = parser.parse_args()

    image = Image.open(args.input_image)
    platerec = Platerec(providers=["CPUExecutionProvider"])
    output = platerec.detect_read(image)

    annotated_image = annotate_images(image, output)
    annotated_image.show()


if __name__ == "__main__":
    main()

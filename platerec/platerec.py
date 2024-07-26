import os
from enum import Enum
from typing import List, Union

import numpy as np
import onnxruntime as ort
import PIL
import PIL.Image
from platerec.config import decode, stoi
from platedet import Platedet


class PlaterecOutputType(Enum):
    WORD = "word"
    CHAR = "char"


class Platerec:
    def __init__(
        self,
        encoder_path: str = os.path.join(
            os.path.dirname(__file__), "artifacts", "encoder.onnx"
        ),
        decoder_path: str = os.path.join(
            os.path.dirname(__file__), "artifacts", "decoder.onnx"
        ),
        use_platedet: bool = True,
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    ):
        self.encoder_session = ort.InferenceSession(
            encoder_path, sess_options=sess_options, providers=providers
        )
        self.decoder_session = ort.InferenceSession(
            decoder_path, sess_options=sess_options, providers=providers
        )
        if use_platedet:
            self.platedet = Platedet(sess_options=sess_options, providers=providers)

    def prepare_input(
        self, image: PIL.Image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ):
        image = image.resize((224, 224))
        image = np.array(image)

        image = image / 255.0
        image = image - np.array(mean)
        image = image / np.array(std)

        image = np.moveaxis(image, -1, 0)
        image = np.expand_dims(image, 0)

        image = image.astype(np.float32)

        return image

    def read(self, image: PIL.Image, max_new_tokens=16, return_type="word"):
        return_type = self._normalize_return_types(return_type)

        image = self.prepare_input(image).astype(np.float16)

        context = self.encoder_session.run(None, {"input": image})[0]
        idx = np.zeros((1, 1), dtype=np.int64)
        probs_log = [1.0]

        for _ in range(max_new_tokens):
            logits = self.decoder_session.run(None, {"input": idx, "context": context})[
                0
            ]
            logits = logits[:, -1, :]

            probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs /= np.sum(probs, axis=-1, keepdims=True)

            idx_next = np.argmax(probs, axis=-1, keepdims=True)
            prob_next = np.max(probs, axis=-1).item()
            probs_log.append(prob_next)

            idx = np.concatenate((idx, idx_next), axis=1)

            if idx_next.item() == stoi[">"]:
                break

        idx = idx.tolist()[0]

        idx = decode(idx[1:-1])
        probs_log = probs_log[1:-1]

        if return_type == PlaterecOutputType.WORD:
            return {"word": idx, "confidence": min(probs_log)}
        elif return_type == PlaterecOutputType.CHAR:
            return {"chars": list(idx), "confidence": probs_log}

        raise ValueError(f"Invalid return type: {return_type}")

    def detect_read(self, image: PIL.Image, max_new_tokens=16, return_type="word"):
        assert self.platedet, "Platedet is not initialized. Use `use_platedet=True` when initializing Platerec."
        words = []
        words_confidences = []

        output = self.platedet.inference(image, return_types=["pil", "boxes"], conf_threshold=0.4)
        if output == {}:
            return {"boxes": [], "words": [], "words_confidences": []}
        
        boxes = output["boxes"]["boxes"]
        output = output["pil"]
        images = output["images"]
        for image in images:
            pred = self.read(
                image, max_new_tokens=max_new_tokens, return_type=return_type
            )
            words.append(pred["word"])
            words_confidences.append(pred["confidence"])

        output["words"] = words
        output["boxes"] = boxes
        output["words_confidences"] = words_confidences

        return output

    def _normalize_return_types(
        self, return_type: Union[PlaterecOutputType, str]
    ) -> List[PlaterecOutputType]:
        if isinstance(return_type, PlaterecOutputType):
            return return_type

        try:
            return PlaterecOutputType(return_type)
        except ValueError:
            raise ValueError(f"Invalid return type: {return_type}")

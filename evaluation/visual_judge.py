"""10-Point Anatomical Visual Rubric Evaluator (Deterministic Spatial + Optional VLM Judge)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import litellm
import numpy as np

from harness.tools import encode_rgb_to_base64_png


@dataclass
class VisualRubricBreakdown:
    """5-part 10-point anatomical visual rubric breakdown."""

    wheels_score: float
    bicycle_frame_score: float
    pelican_head_beak_score: float
    pelican_body_limbs_score: float
    riding_composition_score: float
    total_score: float
    evaluator_type: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize breakdown to dictionary."""
        return asdict(self)


class VisualJudge:
    """Scores the final 1000x500 whiteboard canvas on the 5-part (0..10) pelican-on-bicycle rubric."""

    def __init__(self, vlm_judge_model: str | None = None) -> None:
        self.vlm_judge_model = vlm_judge_model

    def evaluate(self, canvas_rgb: np.ndarray) -> VisualRubricBreakdown:
        """Evaluate canvas_rgb using VLM judge if configured, falling back to deterministic spatial rubric."""
        if self.vlm_judge_model:
            try:
                return self.evaluate_with_vlm(canvas_rgb, self.vlm_judge_model)
            except Exception as exc:  # noqa: BLE001
                _ = exc
        return self.evaluate_deterministic(canvas_rgb)

    def evaluate_deterministic(self, canvas_rgb: np.ndarray) -> VisualRubricBreakdown:
        """Compute deterministic 0..10 visual score from spatial ink distribution across anatomical zones.

        Canvas coordinates (height=500, width=1000):
        - Rows 0..499 map X from 0.300m (top) to 0.160m (bottom).
        - Cols 0..999 map Y from +0.140m (left) to -0.140m (right).
        """
        arr = np.asarray(canvas_rgb, dtype=np.int32)
        # Ink mask: pixels significantly darker than background [250, 250, 252]
        ink_mask = (arr[:, :, 0] < 180) | (arr[:, :, 1] < 180)
        total_ink = int(np.sum(ink_mask))

        if total_ink < 150:
            return VisualRubricBreakdown(
                wheels_score=0.0,
                bicycle_frame_score=0.0,
                pelican_head_beak_score=0.0,
                pelican_body_limbs_score=0.0,
                riding_composition_score=0.0,
                total_score=0.0,
                evaluator_type="deterministic_spatial",
                rationale="Canvas is blank or nearly blank (< 150 ink pixels).",
            )

        h, w = ink_mask.shape

        # 1. Wheels zone: lower half of board (rows 260..485), split into left (cols 120..480) and right (cols 520..880)
        left_wheel_ink = int(np.sum(ink_mask[int(0.50 * h) : int(0.97 * h), int(0.10 * w) : int(0.48 * w)]))
        right_wheel_ink = int(np.sum(ink_mask[int(0.50 * h) : int(0.97 * h), int(0.52 * w) : int(0.90 * w)]))
        wheels_score = 0.0
        if left_wheel_ink >= 350:
            wheels_score += 1.0
        elif left_wheel_ink >= 120:
            wheels_score += 0.5
        if right_wheel_ink >= 350:
            wheels_score += 1.0
        elif right_wheel_ink >= 120:
            wheels_score += 0.5

        # 2. Bicycle frame & handlebars zone: middle-lower bridge connecting wheels (rows 200..400, cols 250..750)
        frame_ink = int(np.sum(ink_mask[int(0.40 * h) : int(0.80 * h), int(0.25 * w) : int(0.75 * w)]))
        if frame_ink >= 900:
            bicycle_frame_score = 2.0
        elif frame_ink >= 400:
            bicycle_frame_score = 1.25
        elif frame_ink >= 150:
            bicycle_frame_score = 0.5
        else:
            bicycle_frame_score = 0.0

        # 3. Pelican body & wings zone: center-upper saddle area (rows 120..300, cols 200..750)
        body_ink = int(np.sum(ink_mask[int(0.24 * h) : int(0.60 * h), int(0.20 * w) : int(0.75 * w)]))
        if body_ink >= 800:
            pelican_body_limbs_score = 2.0
        elif body_ink >= 350:
            pelican_body_limbs_score = 1.25
        elif body_ink >= 120:
            pelican_body_limbs_score = 0.5
        else:
            pelican_body_limbs_score = 0.0

        # 4. Pelican head & pouched beak zone: upper region (rows 20..220, cols 200..850)
        head_beak_ink = int(np.sum(ink_mask[int(0.04 * h) : int(0.44 * h), int(0.20 * w) : int(0.85 * w)]))
        if head_beak_ink >= 650:
            pelican_head_beak_score = 2.0
        elif head_beak_ink >= 250:
            pelican_head_beak_score = 1.25
        elif head_beak_ink >= 80:
            pelican_head_beak_score = 0.5
        else:
            pelican_head_beak_score = 0.0

        # 5. Riding composition & vertical/horizontal balance
        ys, xs = np.where(ink_mask)
        y_span_frac = float(np.percentile(ys, 95) - np.percentile(ys, 5)) / float(h)
        x_span_frac = float(np.percentile(xs, 95) - np.percentile(xs, 5)) / float(w)
        all_four_present = (
            wheels_score >= 1.5
            and bicycle_frame_score >= 1.0
            and pelican_body_limbs_score >= 1.0
            and pelican_head_beak_score >= 1.0
        )
        if all_four_present and y_span_frac >= 0.35 and x_span_frac >= 0.25:
            riding_composition_score = 2.0
        elif y_span_frac >= 0.25 and x_span_frac >= 0.18:
            riding_composition_score = 1.0
        else:
            riding_composition_score = 0.25

        total = round(
            wheels_score
            + bicycle_frame_score
            + pelican_head_beak_score
            + pelican_body_limbs_score
            + riding_composition_score,
            2,
        )
        return VisualRubricBreakdown(
            wheels_score=wheels_score,
            bicycle_frame_score=bicycle_frame_score,
            pelican_head_beak_score=pelican_head_beak_score,
            pelican_body_limbs_score=pelican_body_limbs_score,
            riding_composition_score=riding_composition_score,
            total_score=min(10.0, total),
            evaluator_type="deterministic_spatial",
            rationale=(
                f"Total ink={total_ink}px (wheels={left_wheel_ink}+{right_wheel_ink}px, "
                f"frame={frame_ink}px, body={body_ink}px, head_beak={head_beak_ink}px)."
            ),
        )

    def evaluate_with_vlm(self, canvas_rgb: np.ndarray, judge_model: str) -> VisualRubricBreakdown:
        """Grade canvas_rgb using a Vision-Language Model judge returning JSON rubric scores."""
        b64_png = encode_rgb_to_base64_png(canvas_rgb, max_width=800)
        prompt = (
            "You are grading a robot arm whiteboard drawing of a 'pelican riding a bicycle'. "
            "Score each of these 5 criteria from 0.0 to 2.0 (total out of 10.0) and return ONLY valid JSON:\n"
            "{\n"
            '  "wheels_score": float (0..2, two distinct circular wheels aligned horizontally),\n'
            '  "bicycle_frame_score": float (0..2, connected frame, saddle, and handlebars),\n'
            '  "pelican_head_beak_score": float (0..2, recognizable head and large pouched beak),\n'
            '  "pelican_body_limbs_score": float (0..2, torso, wing, and legs),\n'
            '  "riding_composition_score": float (0..2, pelican properly positioned riding the bicycle),\n'
            '  "rationale": "short 1-sentence explanation"\n'
            "}"
        )
        resp = litellm.completion(
            model=judge_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_png}"},
                        },
                    ],
                }
            ],
            temperature=0.0,
            timeout=60,
        )
        raw_text = (resp.choices[0].message.content or "").strip()
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"VLM Judge did not return JSON: {raw_text}")
        parsed = json.loads(raw_text[start : end + 1])

        w_s = float(np.clip(parsed.get("wheels_score", 0.0), 0.0, 2.0))
        f_s = float(np.clip(parsed.get("bicycle_frame_score", 0.0), 0.0, 2.0))
        h_s = float(np.clip(parsed.get("pelican_head_beak_score", 0.0), 0.0, 2.0))
        b_s = float(np.clip(parsed.get("pelican_body_limbs_score", 0.0), 0.0, 2.0))
        c_s = float(np.clip(parsed.get("riding_composition_score", 0.0), 0.0, 2.0))
        tot = round(w_s + f_s + h_s + b_s + c_s, 2)

        return VisualRubricBreakdown(
            wheels_score=w_s,
            bicycle_frame_score=f_s,
            pelican_head_beak_score=h_s,
            pelican_body_limbs_score=b_s,
            riding_composition_score=c_s,
            total_score=tot,
            evaluator_type=f"vlm_judge:{judge_model}",
            rationale=str(parsed.get("rationale", "")),
        )

"""Strict, self-contained reward replacement for the pinned Time-R1 repo.

Copy this file over the upstream ``reward/reward.py``.  The public entrypoint
keeps the VERL callback signature and returns a finite float.  Forecast syntax
is deliberately strict: a response must contain one reasoning block and one
answer block with an exact, ordered ``1 | value`` table.
"""

from __future__ import annotations

import math
import re
import string
from typing import Any

import numpy as np

HARD_INVALID = -6.0
NMAE_WEIGHT = 0.9
METRIC_CAP = 4.0
TARGET_SCALE_FLOOR_RATIO = 0.05
EPS = 1e-8

_FLOAT_TOKEN = r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?"
_FLOAT_PATTERN = re.compile(rf"\A{_FLOAT_TOKEN}\Z")
_STRUCTURE_PATTERN = re.compile(
    r"\A\s*<think>(?P<think>.*?)</think>\s*"
    r"<answer>(?P<answer>.*?)</answer>\s*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)
_FENCE_PATTERN = re.compile(
    r"\A\s*```[^\r\n`]*\r?\n(?P<body>.*?)\r?\n?```\s*\Z",
    flags=re.DOTALL,
)
_TAG_PATTERNS = {
    "<think>": re.compile(r"<think>", flags=re.IGNORECASE),
    "</think>": re.compile(r"</think>", flags=re.IGNORECASE),
    "<answer>": re.compile(r"<answer>", flags=re.IGNORECASE),
    "</answer>": re.compile(r"</answer>", flags=re.IGNORECASE),
}


class RewardInputError(ValueError):
    """Raised internally when the reward contract is not satisfied."""


def _load_torch_compat():
    try:
        import torch
        import torch.nn as nn
    except (ImportError, OSError) as exc:
        raise ImportError("Time-R1 decomposition compatibility requires PyTorch") from exc
    return torch, nn


class moving_avg:
    """Lazy compatibility constructor for the upstream moving-average module."""

    def __new__(cls, kernel_size, stride):
        torch, nn = _load_torch_compat()

        class _MovingAverage(nn.Module):
            def __init__(self, size, step):
                super().__init__()
                self.kernel_size = size
                self.avg = nn.AvgPool1d(kernel_size=size, stride=step, padding=0)

            def forward(self, x):
                front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
                end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
                padded = torch.cat([front, x, end], dim=1)
                return self.avg(padded.permute(0, 2, 1)).permute(0, 2, 1)

        return _MovingAverage(kernel_size, stride)


class series_decomp:
    """Lazy compatibility constructor for the upstream decomposition module."""

    def __new__(cls, kernel_size):
        _, nn = _load_torch_compat()

        class _SeriesDecomp(nn.Module):
            def __init__(self, size):
                super().__init__()
                self.moving_avg = moving_avg(size, stride=1)

            def forward(self, x):
                moving_mean = self.moving_avg(x)
                return x - moving_mean, moving_mean

        return _SeriesDecomp(kernel_size)


def _invalid(reason: str) -> RewardInputError:
    return RewardInputError(reason)


def _as_finite_sequence(values: Any, *, field: str) -> list[float]:
    if isinstance(values, str):
        text = values.strip()
        if not text:
            raise _invalid(f"{field}_empty")
        tokens: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [part for part in re.split(r"[\s,]+", line) if part]
            if not parts or any(_FLOAT_PATTERN.fullmatch(part) is None for part in parts):
                raise _invalid(f"{field}_malformed_text")
            tokens.extend(parts)
        if not tokens:
            raise _invalid(f"{field}_empty")
        try:
            array = np.asarray([float(token) for token in tokens], dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise _invalid(f"{field}_not_numeric") from exc
    else:
        try:
            array = np.asarray(values, dtype=float)
            if array.ndim == 0:
                array = array.reshape(1)
            elif array.ndim != 1:
                raise _invalid(f"{field}_not_1d")
        except (TypeError, ValueError, OverflowError) as exc:
            raise _invalid(f"{field}_not_numeric") from exc

    if array.size == 0 or not np.all(np.isfinite(array)):
        raise _invalid(f"{field}_non_finite_or_empty")
    return [float(value) for value in array.tolist()]


def _parse_ground_truth(values: Any) -> list[float]:
    return _as_finite_sequence(values, field="ground_truth")


def extract_ground_truth_values(text: Any) -> list[float]:
    """Return finite GT values, or an empty list for an invalid GT payload."""

    try:
        return _parse_ground_truth(text)
    except (RewardInputError, TypeError, ValueError, OverflowError):
        return []


def _split_cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_header(line: str) -> bool:
    cells = _split_cells(line)
    if len(cells) != 2:
        return False
    return cells[0].lower() in {"index", "step", "t"} and cells[1].lower() in {
        "value",
        "forecast",
        "prediction",
    }


def _is_separator(line: str) -> bool:
    cells = _split_cells(line)
    return len(cells) == 2 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _parse_structure(text: Any) -> str:
    if text is None:
        raise _invalid("empty_response")
    response = str(text)
    if not response.strip():
        raise _invalid("empty_response")
    if any(len(pattern.findall(response)) != 1 for pattern in _TAG_PATTERNS.values()):
        raise _invalid("structural_tag_count")
    match = _STRUCTURE_PATTERN.fullmatch(response)
    if match is None:
        raise _invalid("structural_block_order")
    if not match.group("think").strip():
        raise _invalid("empty_think_block")
    return match.group("answer")


def _parse_forecast(text: Any, *, expected_length: int | None) -> list[float]:
    answer = _parse_structure(text)
    if answer.count("```") != 2:
        raise _invalid("code_fence_count")
    fence = _FENCE_PATTERN.fullmatch(answer)
    if fence is None:
        raise _invalid("malformed_code_fence")

    indices: list[int] = []
    values: list[float] = []
    for raw_line in fence.group("body").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _is_header(line):
            if values:
                raise _invalid("header_after_data")
            continue
        if _is_separator(line):
            if values:
                raise _invalid("separator_after_data")
            continue

        cells = _split_cells(line)
        if len(cells) != 2 or re.fullmatch(r"[0-9]+", cells[0]) is None:
            raise _invalid("non_table_content")
        if _FLOAT_PATTERN.fullmatch(cells[1]) is None:
            raise _invalid("non_numeric_table_value")
        try:
            index = int(cells[0])
            value = float(cells[1])
        except (TypeError, ValueError, OverflowError) as exc:
            raise _invalid("non_numeric_table_value") from exc
        if cells[0] != str(index):
            raise _invalid("non_canonical_index")
        if not math.isfinite(value):
            raise _invalid("non_finite_prediction")
        indices.append(index)
        values.append(value)

    if not values:
        raise _invalid("empty_forecast")
    if expected_length is None:
        expected_indices = list(range(1, len(values) + 1))
    else:
        if expected_length <= 0:
            raise _invalid("invalid_expected_length")
        expected_indices = list(range(1, expected_length + 1))
    if indices != expected_indices:
        raise _invalid("answer_indices_or_length")
    return values


def parse_forecast_answer(text: Any, *, expected_length: int | None = None) -> list[float]:
    """Parse a response under the strict forecast contract.

    The helper raises ``RewardInputError`` so callers that need a rejection
    reason can inspect it; the public reward callback catches it and returns
    ``HARD_INVALID`` instead of allowing an exception into the trainer.
    """

    return _parse_forecast(text, expected_length=expected_length)


def extract_answer(text: Any) -> str:
    """Compatibility helper returning the answer block when one exists."""

    match = re.search(r"<answer>(.*?)</answer>", str(text), flags=re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else str(text)


def extract_solution_answer(solution_str: Any) -> str | None:
    match = re.search(
        r"<answer>(.*?)</answer>",
        str(solution_str),
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else None


def extract_qwen_results(text: Any, attr: str | None = None) -> list[float]:
    """Compatibility helper that returns only a fully valid forecast table."""

    del attr
    try:
        return parse_forecast_answer(text)
    except (RewardInputError, TypeError, ValueError, OverflowError):
        return []


def compute_format_score(solution_str: Any) -> float:
    try:
        parse_forecast_answer(solution_str)
    except (RewardInputError, TypeError, ValueError, OverflowError):
        return -1.0
    return 0.0


def _safe_scaled_product(reference: float, factor: float) -> float:
    if reference == 0.0 or factor <= 0.0:
        return 0.0
    max_float = float(np.finfo(float).max)
    if factor >= max_float / reference:
        return max_float
    return reference * factor


def _target_center(target: np.ndarray) -> float:
    reference = float(np.max(np.abs(target)))
    if reference == 0.0:
        return 0.0
    scaled_center = float(np.mean(target / reference))
    if scaled_center >= 0.0:
        return _safe_scaled_product(reference, scaled_center)
    return -_safe_scaled_product(reference, -scaled_center)


def _target_scale(target: np.ndarray) -> float:
    # Scale first so mean/std stay finite even when the original values are near
    # the largest representable float.
    reference = float(np.max(np.abs(target)))
    if reference == 0.0:
        return EPS
    scaled = target / reference
    level_factor = float(np.mean(np.abs(scaled)))
    spread_factor = float(np.std(scaled))
    scale = max(
        _safe_scaled_product(reference, spread_factor),
        TARGET_SCALE_FLOOR_RATIO * _safe_scaled_product(reference, level_factor),
        EPS,
    )
    if not math.isfinite(scale):
        raise _invalid("non_finite_target_scale")
    return scale


def reward_norm(x_list: Any, y_list: Any) -> tuple[list[float], list[float]]:
    """Normalize predictions and GT using only the GT mean and scale."""

    try:
        prediction = np.asarray(_as_finite_sequence(x_list, field="prediction"), dtype=float)
        target = np.asarray(_parse_ground_truth(y_list), dtype=float)
        scale = _target_scale(target)
        center = _target_center(target)
        normalized_prediction = (prediction - center) / scale
        normalized_target = (target - center) / scale
        if not np.all(np.isfinite(normalized_prediction)):
            raise _invalid("non_finite_normalized_prediction")
        return normalized_prediction.tolist(), normalized_target.tolist()
    except (RewardInputError, TypeError, ValueError, OverflowError, FloatingPointError):
        return [], []


# Keep the misspelled upstream helper name usable for callers that imported it.
reworad_norm = reward_norm


def mean_squared_error(y_true: Any, y_pred: Any) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.square(true - pred)))


def decompose(values: Any) -> tuple[np.ndarray, np.ndarray]:
    """Numerically safe moving-average decomposition for legacy callers."""

    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return array.copy(), array.copy()
    kernel_size = 25
    pad = (kernel_size - 1) // 2
    padded = np.concatenate(
        [np.repeat(array[:1], pad), array, np.repeat(array[-1:], pad)]
    )
    kernel = np.ones(kernel_size, dtype=float) / kernel_size
    trend = np.convolve(padded, kernel, mode="valid")
    return array - trend, trend


def mean_squared_error_season_trend(y_true: Any, y_pred: Any) -> tuple[float, float]:
    season_true, trend_true = decompose(y_true)
    season_pred, trend_pred = decompose(y_pred)
    return mean_squared_error(season_true, season_pred), mean_squared_error(trend_true, trend_pred)


def _score_forecast(prediction: list[float], target: list[float]) -> float:
    if len(prediction) != len(target) or not prediction:
        raise _invalid("answer_indices_or_length")
    pred = np.asarray(prediction, dtype=float)
    gt = np.asarray(target, dtype=float)
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(gt)):
        raise _invalid("non_finite_metric_input")
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        scale = _target_scale(gt)
        # Work in the normalized domain so finite opposite-sign extremes do
        # not overflow during subtraction or when squaring the raw error.
        normalized_error = (pred / scale) - (gt / scale)
        nmae = float(np.mean(np.abs(normalized_error)))
        nmse = float(np.mean(np.square(normalized_error)))
    if not math.isfinite(nmae) or not math.isfinite(nmse):
        raise _invalid("non_finite_metric")
    metric_loss = NMAE_WEIGHT * min(max(nmae, 0.0), METRIC_CAP)
    metric_loss += (1.0 - NMAE_WEIGHT) * min(max(nmse, 0.0), METRIC_CAP)
    score = 1.0 - metric_loss
    return float(score) if math.isfinite(score) else HARD_INVALID


def compute_answer_score(solution_str: Any, ground_truth: Any) -> float:
    try:
        target = _parse_ground_truth(ground_truth)
        prediction = _parse_forecast(solution_str, expected_length=len(target))
        return _score_forecast(prediction, target)
    except (RewardInputError, TypeError, ValueError, OverflowError, FloatingPointError):
        return HARD_INVALID


def compute_score_length(solution_str: Any, ground_truth: Any) -> float:
    """Compatibility helper; partial answers receive no length reward."""

    try:
        target = _parse_ground_truth(ground_truth)
        _parse_forecast(solution_str, expected_length=len(target))
    except (RewardInputError, TypeError, ValueError, OverflowError):
        return 0.0
    return 0.1


def compute_answer_score_season_trend(solution_str: Any, ground_truth: Any) -> float:
    """Legacy diagnostic retained for imports, but excluded from training score."""

    try:
        target = _parse_ground_truth(ground_truth)
        prediction = _parse_forecast(solution_str, expected_length=len(target))
        norm_prediction, norm_target = reward_norm(prediction, target)
        if not norm_prediction or not norm_target:
            return 0.0
        season_mse, trend_mse = mean_squared_error_season_trend(
            norm_target,
            norm_prediction,
        )
        value = 0.5 * (season_mse + trend_mse)
        return float(1.0 / (1.0 + max(value, 0.0))) if math.isfinite(value) else 0.0
    except (RewardInputError, TypeError, ValueError, OverflowError, FloatingPointError):
        return 0.0


def change_point(data: Any) -> tuple[list[int], list[int]]:
    """Return safe local extrema indices; short sequences produce no points."""

    try:
        values = [float(value) for value in np.asarray(data, dtype=float).reshape(-1)]
    except (TypeError, ValueError, OverflowError):
        return [], []
    if len(values) < 3 or not all(math.isfinite(value) for value in values):
        return [], []

    maxima: list[int] = []
    minima: list[int] = []
    for index, center in enumerate(values):
        neighbors = values[max(0, index - 2) : index] + values[index + 1 : index + 3]
        if not neighbors:
            continue
        if all(center >= value for value in neighbors) and any(center > value for value in neighbors):
            maxima.append(index)
        if all(center <= value for value in neighbors) and any(center < value for value in neighbors):
            minima.append(index)
    return maxima, minima


def _f1(predicted: list[int], actual: list[int]) -> float:
    predicted_set = set(predicted)
    actual_set = set(actual)
    if not predicted_set and not actual_set:
        return 1.0
    if not predicted_set or not actual_set:
        return 0.0
    true_positive = len(predicted_set & actual_set)
    precision = true_positive / len(predicted_set)
    recall = true_positive / len(actual_set)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def conmute_change_point(solution_str: Any, ground_truth: Any) -> float:
    """Compatibility shape metric with false-positive and short-input safety."""

    try:
        target = _parse_ground_truth(ground_truth)
        prediction = _parse_forecast(solution_str, expected_length=len(target))
        target_max, target_min = change_point(target)
        prediction_max, prediction_min = change_point(prediction)
        return float(0.1 * (_f1(prediction_max, target_max) + _f1(prediction_min, target_min)))
    except (RewardInputError, TypeError, ValueError, OverflowError):
        return 0.0


def normalize_answer(text: Any) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        return "".join(ch for ch in value if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(str(text).lower())))


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """VERL-compatible reward callback.

    ``data_source`` and ``extra_info`` are accepted for API compatibility.  A
    malformed or non-finite sample always receives the fixed floor and never
    contributes a partial length/shape reward.
    """

    del data_source, extra_info
    try:
        target = _parse_ground_truth(ground_truth)
        prediction = _parse_forecast(solution_str, expected_length=len(target))
        return _score_forecast(prediction, target)
    except (RewardInputError, TypeError, ValueError, OverflowError, FloatingPointError):
        return float(HARD_INVALID)

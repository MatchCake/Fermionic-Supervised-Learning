from typing import Optional

import numpy as np
from numpy._typing import NDArray
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler


X_SCALINGS = ("none", "minmax01", "minmax11", "standard", "quantile")
MAX_QUANTILE_KNOTS = 1000
QUANTILE_RANDOM_STATE = 0


def make_x_scaler(x_scaling: Optional[str], n_samples: int) -> Optional[TransformerMixin]:
    """Build the input transform named by ``x_scaling``.

    ``"quantile"`` is the one that is not a rescaling: it maps each feature onto its own rank,
    so a heavy tail stops compressing the rest of the column. That matters here because the
    matchgate embedding turns a feature into a rotation angle, and after a min-max scaling of
    California Housing 99% of ``AveOccup`` lands inside ``[0, 0.0038]``: the angle barely moves
    and the feature is effectively invisible to the circuit. Three of the other seven columns
    are squashed the same way.

    :param x_scaling: One of :data:`X_SCALINGS`, or None which is equivalent to ``"none"``.
    :param n_samples: Number of training samples, used to clamp the quantile knot count.
    :return: An unfitted transformer, or None when the inputs are passed through untouched.
    :rtype: Optional[TransformerMixin]
    :raises ValueError: If ``x_scaling`` is not a recognised name.
    """
    if x_scaling is None or x_scaling == "none":
        return None
    if x_scaling == "minmax01":
        return MinMaxScaler(feature_range=(0.0, 1.0))
    if x_scaling == "minmax11":
        return MinMaxScaler(feature_range=(-1.0, 1.0))
    if x_scaling == "standard":
        return StandardScaler()
    if x_scaling == "quantile":
        return QuantileTransformer(
            output_distribution="uniform",
            n_quantiles=max(2, min(MAX_QUANTILE_KNOTS, n_samples)),
            random_state=QUANTILE_RANDOM_STATE,
        )
    raise ValueError(f"Unknown x_scaling {x_scaling!r}; expected one of {X_SCALINGS} or None.")


class InputScalingMixin:
    """Fit an input transform on the training split and reapply it at predict time.

    The scaler is fitted on ``x_train`` alone, so it carries no validation or test information,
    and the same fitted transform is applied to every later input. A test point outside the
    training range therefore lands outside the transformed window rather than being renormalised
    against its own batch, which is the correct behaviour and also why :class:`TargetRangeClipMixin`
    exists.

    Consumers set ``x_scaling`` as a keyword-only constructor argument and call
    :meth:`_fit_x_scaler` from ``fit`` and :meth:`_scale_x` from ``predict``.
    """

    # Declared, not assigned: the consuming estimator sets x_scaling in its own __init__, so a
    # class-level value here would shadow it and hide it from sklearn's get_params.
    x_scaling: Optional[str]
    x_scaler_: Optional[TransformerMixin]

    def _fit_x_scaler(self, x_train: NDArray) -> NDArray:
        """Fit the input transform on the training split and return the transformed inputs.

        :param x_train: Training inputs of shape ``(n_samples, n_features)``.
        :return: The transformed inputs, or the originals when scaling is disabled.
        :rtype: NDArray
        """
        x_train = np.asarray(x_train)
        self.x_scaler_ = make_x_scaler(self.x_scaling, len(x_train))
        if self.x_scaler_ is None:
            return x_train
        scaled = self.x_scaler_.fit_transform(x_train.astype(np.float64))
        return scaled.astype(x_train.dtype, copy=False)

    def _scale_x(self, x: NDArray) -> NDArray:
        """Apply the fitted input transform.

        :param x: Inputs of shape ``(n_samples, n_features)``.
        :return: The transformed inputs, or the originals when scaling is disabled.
        :rtype: NDArray
        """
        x = np.asarray(x)
        if self.x_scaler_ is None:
            return x
        scaled = self.x_scaler_.transform(x.astype(np.float64))
        return scaled.astype(x.dtype, copy=False)


class TargetRangeClipMixin:
    """Clip predictions to the range of targets seen during training.

    Consumers call :meth:`_fit_target_range` from ``fit`` and :meth:`_clip_to_target_range` from
    ``predict``.
    """

    y_train_min_: float
    y_train_max_: float

    def _fit_target_range(self, y_train: NDArray) -> None:
        """Record the training target range used to bound later predictions.

        :param y_train: Training targets of shape ``(n_samples,)``.
        """
        y_train = np.asarray(y_train, dtype=np.float64)
        self.y_train_min_ = float(np.min(y_train))
        self.y_train_max_ = float(np.max(y_train))

    def _clip_to_target_range(self, predictions: NDArray) -> NDArray:
        """Clip predictions into the recorded training target range.

        :param predictions: Predicted targets of shape ``(n_samples,)``.
        :return: The clipped predictions.
        :rtype: NDArray
        """
        return np.clip(predictions, self.y_train_min_, self.y_train_max_)

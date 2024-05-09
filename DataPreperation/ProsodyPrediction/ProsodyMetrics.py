import tensorflow as tf


class AbsoluteAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):
    """
    The class that calculate the Traditional Accuracy of the model the input is Logits, Y is internal value
    """

    def __init__(self, name='absolute_accuracy', dtype=None):
        super(AbsoluteAccuracy, self).__init__(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update the state of the accuracy
        :param y_true: The true label in integer
        :param y_pred: The predicted label in logits with dimension [batch_size, num_classes]
        :param sample_weight: The weight of the sample
        :return: None
        """
        # y_abs_label, _, _ = y_true
        y_true = tf.math.round(y_true[:, 0] / 4 - 0.15)
        y_true = tf.cast(y_true, tf.int32)
        super(AbsoluteAccuracy, self).update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        config = super(AbsoluteAccuracy, self).get_config()
        return config

    def reset_state(self):
        super(AbsoluteAccuracy, self).reset_state()
        return


class WithInRangeAccuracy(tf.keras.metrics.Metric):
    """
    The class that calculate the accuracy of the model within the range of the true label
    1. Calculate the probability of the predicted label by using the softmax function
    2. Calculate the Expected Value of the predicted label by using the probability and score 0 ~ 3
    3. Compare the Expected Value with the true label to check if the predicted label is within the range of the true label
    4. Return the accuracy
    """

    def __init__(self, name='within_range_accuracy', dtype=None):
        # Generate a mean metric
        self.mean = tf.keras.metrics.Mean(name='mean')
        super().__init__(name=name, dtype=dtype)

    def update_state(self, y_true_packed, y_pred, sample_weight=None):
        _, y_min, y_max = tf.unstack(y_true_packed, axis=1)
        y_min = tf.math.round(y_min - 0.15)
        y_max = tf.math.round(y_max - 0.15)
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        liner_space = tf.cast(tf.linspace(0, 3, 4), tf.float32)
        y_pred = tf.reduce_sum(liner_space * y_pred, axis=-1)
        # if y_min <= y_pred <= y_max it is within the range and considered as correct else incorrect
        within_range = tf.logical_and(y_pred >= y_min - 0.25, y_pred <= y_max + 0.75)
        # Return the accuracy
        self.mean.update_state(within_range, sample_weight)

    def result(self):
        return self.mean.result()

    def reset_state(self):
        self.mean.reset_state()
        return

    def get_config(self):
        config = super(WithInRangeAccuracy, self).get_config()
        return config


class RatingAccuracy(tf.keras.metrics.Metric):
    """
    The class calculate the Rating Accuracy of the model which needs to match up the rating rubric

    1. The value of y_pred is a logits value with total 13 dimensions from score 0 to 3 with 0.25 interval
    2. The y_true is a interger value from 0 to 12 that represents rating
    3. the rubic defined as follows if score is x.0 ~ x. 75(not included) the rating is x x.75 ~ x+1 the rating is x+1
    """

    def __init__(self, name='rating_accuracy', dtype=None):
        self.mean = tf.keras.metrics.Mean(name='mean')
        super().__init__(name=name, dtype=dtype)

    def update_state(self, y_true_packed, y_pred, sample_weight=None):
        y_true = tf.math.round(y_true_packed[:, 0] / 4 - 0.15)
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        liner_space = tf.cast(tf.linspace(0, 3, 4), tf.float32)
        y_pred = tf.reduce_sum(liner_space * y_pred, axis=-1)
        # round is 0.5 split we need 0.75 split 0.6 should be 0, 0.8 should be 1. So that we need to offset 0.25
        # y_pred = y_pred - 0.25
        # y_pred = tf.math.round(y_pred)
        # y_true = tf.cast(y_true, tf.float32) / 4 - 0.25
        # y_true = tf.math.round(y_true)
        y_pred = tf.cast(y_pred, tf.int32)
        correct = tf.equal(y_pred, y_true)
        self.mean.update_state(correct, sample_weight)

    def result(self):
        return self.mean.result()

    def reset_state(self):
        self.mean.reset_state()
        return

    def get_config(self):
        config = super(RatingAccuracy, self).get_config()
        return config


class RatingAccuracyDirect(RatingAccuracy):
    def __init__(self, name='rating_accuracy_direct', dtype=None):
        super().__init__(name=name, dtype=dtype)

    def update_state(self, y_true_packed, y_pred, sample_weight=None):
        # Using Argmax to get the rating
        y_true = tf.math.round(y_true_packed[:, 0] / 4 - 0.15)
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = tf.cast(y_pred, tf.int32)
        # y_pred = tf.cast(y_pred, tf.float32) / 4 - 0.15
        # y_pred = tf.math.round(y_pred)
        # y_true = tf.cast(y_true, tf.float32) / 4 - 0.15
        # y_true = tf.math.round(y_true)
        correct = tf.equal(y_pred, y_true)
        self.mean.update_state(correct, sample_weight)


class WithInRangeAccuracyDirect(WithInRangeAccuracy):
    def __init__(self, name='within_range_accuracy_direct', dtype=None):
        super().__init__(name=name, dtype=dtype)

    def update_state(self, y_true_packed, y_pred, sample_weight=None):
        _, y_min, y_max = tf.unstack(y_true_packed, axis=1)
        y_min = tf.math.round(y_min - 0.15)
        y_max = tf.math.round(y_max - 0.15)
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = tf.cast(y_pred, tf.float32)
        within_range = tf.logical_and(y_pred >= (y_min - 0.25), y_pred <= (y_max + 0.75))
        self.mean.update_state(within_range, sample_weight)


# Define a Keras Model Loss
class ProsodyLoss(tf.keras.losses.Loss):
    def __init__(self, factor=1, name='prosody_loss', **kwargs):
        """
        The Prosody Loss is a combination of Mean Squared Error and Categorical Cross Entropy
        :param factor: default 1. The factor to divide the loss represents GPU counts to scale the loss to correct value
        """
        super().__init__(name=name, **kwargs)
        self.MSESubLoss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.CCELoss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,
                                                                     reduction=tf.keras.losses.Reduction.NONE)
        # self.ABLoss = tf.Variable([0, 0], trainable=True, dtype=tf.float32, name='absolute_loss')
        self.factor = factor

    @tf.function
    def call(self, y_true_packed, y_pred):
        y_true = tf.math.round(y_true_packed[:, 0] / 4 - 0.15)
        y_true = tf.cast(y_true, tf.int32)
        y_prob = tf.nn.softmax(y_pred, axis=-1)
        liner_space = tf.linspace(0, 3, 4)
        y_expected = tf.reduce_sum(tf.cast(liner_space, tf.float32) * y_prob, axis=-1, keepdims=True)
        y_true_scaled = tf.cast(y_true, tf.float32)[..., tf.newaxis]
        y_loss_mse = self.MSESubLoss(y_true_scaled, y_expected)
        y_loss_cce = self.CCELoss(y_true, y_prob)
        # # Weight loss by ABL weight
        # weight_exp = 1 / (tf.exp(self.ABLoss) ** 2)
        #
        # # Combine y_loss_mse and y_loss_cce in to a vector shape [batch_size, 2]
        # y_loss = tf.stack([2 * y_loss_mse, y_loss_cce], axis=-1)
        # y_weighted_loss = tf.reduce_sum(y_loss * weight_exp, axis=-1)
        # # return average loss over GPU mini batch size divided by factor
        # return tf.reduce_mean(y_weighted_loss) / self.factor
        return (1e-4 * tf.reduce_mean(y_loss_mse) + tf.reduce_mean(y_loss_cce)) / self.factor

    def get_config(self):
        config = super(ProsodyLoss, self).get_config()
        config.update({'factor': self.factor})
        return config


class WeightedKappaLoss(tf.keras.losses.Loss):

    def __init__(
            self,
            num_classes: int,
            weightage="quadratic",
            name="cohen_kappa_loss",
            epsilon=1e-6,
            reduction: str = tf.keras.losses.Reduction.NONE,
    ):
        r"""Creates a `WeightedKappaLoss` instance.

        Args:
          num_classes: Number of unique classes in your dataset.
          weightage: (Optional) Weighting to be considered for calculating
            kappa statistics. A valid value is one of
            ['linear', 'quadratic']. Defaults to 'quadratic'.
          name: (Optional) String name of the metric instance.
          epsilon: (Optional) increment to avoid log zero,
            so the loss will be $ \log(1 - k + \epsilon) $, where $ k $ lies
            in $ [-1, 1] $. Defaults to 1e-6.
        Raises:
          ValueError: If the value passed for `weightage` is invalid
            i.e. not any one of ['linear', 'quadratic']
        """

        super().__init__(name=name, reduction=reduction)

        if weightage not in ("linear", "quadratic"):
            raise ValueError("Unknown kappa weighting type.")

        self.weightage = weightage
        self.num_classes = num_classes
        self.epsilon = epsilon or tf.keras.backend.epsilon()
        label_vec = tf.range(num_classes, dtype=tf.keras.backend.floatx())
        self.row_label_vec = tf.reshape(label_vec, [1, num_classes])
        self.col_label_vec = tf.reshape(label_vec, [num_classes, 1])
        col_mat = tf.tile(self.col_label_vec, [1, num_classes])
        row_mat = tf.tile(self.row_label_vec, [num_classes, 1])
        if weightage == "linear":
            self.weight_mat = tf.abs(col_mat - row_mat)
        else:
            self.weight_mat = (col_mat - row_mat) ** 2

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=self.col_label_vec.dtype)
        y_pred = tf.cast(y_pred, dtype=self.weight_mat.dtype)
        batch_size = tf.shape(y_true)[0]
        cat_labels = tf.matmul(y_true, self.col_label_vec)
        cat_label_mat = tf.tile(cat_labels, [1, self.num_classes])
        row_label_mat = tf.tile(self.row_label_vec, [batch_size, 1])
        if self.weightage == "linear":
            weight = tf.abs(cat_label_mat - row_label_mat)
        else:
            weight = (cat_label_mat - row_label_mat) ** 2
        numerator = tf.reduce_sum(weight * y_pred)
        label_dist = tf.reduce_sum(y_true, axis=0, keepdims=True)
        pred_dist = tf.reduce_sum(y_pred, axis=0, keepdims=True)
        w_pred_dist = tf.matmul(self.weight_mat, pred_dist, transpose_b=True)
        denominator = tf.reduce_sum(tf.matmul(label_dist, w_pred_dist))
        denominator /= tf.cast(batch_size, dtype=denominator.dtype)
        loss = tf.math.divide_no_nan(numerator, denominator)
        return tf.math.log(loss + self.epsilon)

    def get_config(self):
        config = {
            "num_classes": self.num_classes,
            "weightage": self.weightage,
            "epsilon": self.epsilon,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class ProsodyKappaLoss(WeightedKappaLoss):
    def __init__(self, factor=1, num_classes=4, name='prosody_kappa_loss', **kwargs):
        super().__init__(num_classes=num_classes, weightage="quadratic", name=name, epsilon=1e-6, reduction=tf.keras.losses.Reduction.NONE, **kwargs)
        # Factor is a GPU count to scale the loss to correct value
        self.factor = factor

    def call(self, y_packed, y_pred):
        y_true = tf.math.round(y_packed[:, 0] / 4 - 0.15)
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, 4, axis=-1)
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        kappa_loss = super().call(y_true, y_pred)
        return tf.reduce_sum(kappa_loss) / self.factor

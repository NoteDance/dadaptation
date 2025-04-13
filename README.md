# DAdaptAdam

**Overview**:

The `DAdaptAdam` optimizer is an adaptive optimization algorithm that dynamically adjusts its update scaling based on observed gradient statistics. It extends the Adam framework by introducing a separate scaling accumulator and an adaptive scaling factor (d₀) that evolves during training. This dynamic adaptation enables DAdaptAdam to automatically calibrate the effective learning rate based on the structure and variability of gradients. Additional features such as optional bias correction, decoupled weight decay, and fixed decay further enhance its robustness in diverse training scenarios.

**Parameters**:

- **`learning_rate`** *(float, default=1.0)*: Base learning rate for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment (mean) estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment (variance) estimates.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay. Applied either in a decoupled fashion or directly, depending on `weight_decouple`.
- **`d0`** *(float, default=1e-6)*: Initial scaling factor that adapts the update magnitude based on gradient statistics.
- **`growth_rate`** *(float, default=`inf`)*: Upper bound on the allowed growth of the scaling factor.
- **`weight_decouple`** *(bool, default=True)*: Whether to decouple weight decay from the gradient update.
- **`fixed_decay`** *(bool, default=False)*: Uses a fixed weight decay value instead of scaling it by the learning rate.
- **`bias_correction`** *(bool, default=False)*: If enabled, applies bias correction when computing the adaptive scaling factor.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by the global norm across all parameters.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum coefficient for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) for updating EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before an update.
- **`name`** *(str, default="dadaptadam")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from dadaptadam import DAdaptAdam

# Instantiate the DAdaptAdam optimizer
optimizer = DAdaptAdam(
    learning_rate=1.0,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=0.0,
    d0=1e-6,
    growth_rate=float('inf'),
    weight_decouple=True,
    fixed_decay=False,
    bias_correction=False
)

# Compile a model with the DAdaptAdam optimizer
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# DAdaptSGD

**Overview**:

The `DAdaptSGD` optimizer is an adaptive variant of stochastic gradient descent that automatically calibrates its effective learning rate based on the observed gradient statistics. By maintaining an accumulated statistic of gradients and leveraging a dynamic scaling factor (d₀), it adjusts updates to better match the curvature of the loss landscape. Additionally, the optimizer supports momentum and decoupled weight decay, making it a robust choice for training deep neural networks with varying gradient scales.

**Parameters**:

- **`learning_rate`** *(float, default=1.0)*: The base learning rate for parameter updates.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay regularization. When non-zero, weight decay is applied either decoupled from the gradient update or directly, based on `weight_decouple`.
- **`momentum`** *(float, default=0.9)*: Momentum factor for smoothing updates.
- **`d0`** *(float, default=1e-6)*: Initial scaling factor that adapts the effective learning rate based on accumulated gradient information.
- **`growth_rate`** *(float, default=`inf`)*: Maximum factor by which the scaling factor is allowed to grow during training.
- **`weight_decouple`** *(bool, default=True)*: Determines whether weight decay is applied decoupled from the gradient update.
- **`fixed_decay`** *(bool, default=False)*: If enabled, applies a fixed weight decay rather than scaling it by the learning rate.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by the global norm across all parameters.
- **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Steps for accumulating gradients before performing an update.
- **`name`** *(str, default="dadaptsgd")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from dadaptsgd import DAdaptSGD

# Instantiate the DAdaptSGD optimizer
optimizer = DAdaptSGD(
    learning_rate=1.0,
    weight_decay=1e-2,
    momentum=0.9,
    d0=1e-6,
    growth_rate=float('inf'),
    weight_decouple=True,
    fixed_decay=False
)

# Compile a model with the DAdaptSGD optimizer
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# DAdaptLion

**Overview**:

The `DAdaptLion` optimizer is an adaptive variant of the Lion optimizer that dynamically adjusts its scaling factor based on accumulated gradient statistics. It uses a sign-based update rule combined with an exponential moving average and a secondary accumulator to compute a dynamic scaling parameter (d₀). This mechanism allows the optimizer to automatically calibrate its effective learning rate in response to the observed gradient structure while optionally applying decoupled weight decay.

**Parameters**:

- **`learning_rate`** *(float, default=1.0)*: The base learning rate for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimate used in the sign update.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for accumulating gradient statistics.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay. When non-zero, weight decay is applied either decoupled from the update or directly, depending on `weight_decouple`.
- **`d0`** *(float, default=1e-6)*: Initial adaptive scaling factor that adjusts the effective step size based on gradient statistics.
- **`weight_decouple`** *(bool, default=True)*: If True, applies weight decay decoupled from the gradient update.
- **`fixed_decay`** *(bool, default=False)*: Uses a fixed weight decay value instead of scaling it by the learning rate.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by the global norm across all parameters.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum coefficient for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) for updating EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before performing an update.
- **`name`** *(str, default="dadaptlion")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from dadaptlion import DAdaptLion

# Instantiate the DAdaptLion optimizer
optimizer = DAdaptLion(
    learning_rate=1.0,
    beta1=0.9,
    beta2=0.999,
    weight_decay=1e-2,
    d0=1e-6,
    weight_decouple=True,
    fixed_decay=False
)

# Compile a TensorFlow/Keras model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# DAdaptAdan

**Overview**:

The `DAdaptAdan` optimizer is an adaptive optimization algorithm that extends the Adam family by dynamically adjusting its effective learning rate based on observed gradient differences and higher-order statistics. By maintaining separate exponential moving averages for the gradients, their squared values, and their differences, it computes an adaptive scaling factor (d₀) that automatically calibrates the update magnitude during training. This approach aims to improve convergence and robustness, especially in scenarios with varying gradient dynamics. Additionally, the optimizer supports decoupled weight decay and flexible decay scaling.

**Parameters**:

- **`learning_rate`** *(float, default=1.0)*: The base learning rate for parameter updates.
- **`beta1`** *(float, default=0.98)*: Exponential decay rate for the first moment (mean) estimates.
- **`beta2`** *(float, default=0.92)*: Exponential decay rate for the moving average of gradient differences.
- **`beta3`** *(float, default=0.99)*: Exponential decay rate for the second moment (variance) estimates of the gradient differences.
- **`epsilon`** *(float, default=1e-8)*: Small constant added for numerical stability.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay. When non-zero, weight decay is applied either in a decoupled manner or directly to the gradients.
- **`d0`** *(float, default=1e-6)*: Initial adaptive scaling factor that governs the effective update magnitude.
- **`growth_rate`** *(float, default=`inf`)*: Upper bound for the allowed growth of the scaling factor during training.
- **`weight_decouple`** *(bool, default=True)*: If True, decouples weight decay from the gradient update.
- **`fixed_decay`** *(bool, default=False)*: Uses fixed weight decay instead of scaling it by the learning rate.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by the global norm across all parameters.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average (EMA) to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for updating EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before performing an update.
- **`name`** *(str, default="dadaptadan")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from dadaptadan import DAdaptAdan

# Instantiate the DAdaptAdan optimizer
optimizer = DAdaptAdan(
    learning_rate=1.0,
    beta1=0.98,
    beta2=0.92,
    beta3=0.99,
    epsilon=1e-8,
    weight_decay=1e-2,
    d0=1e-6,
    growth_rate=float('inf'),
    weight_decouple=True,
    fixed_decay=False
)

# Compile a model using the DAdaptAdan optimizer
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# DAdaptAdaGrad

**Overview**:

The `DAdaptAdaGrad` optimizer is an adaptive optimization algorithm that builds upon the AdaGrad method by dynamically adjusting its effective update scaling. It maintains per-parameter accumulators for squared gradients (stored in `alpha_k`), an auxiliary accumulator `sk` for gradient updates, and the initial parameter values `x0`. These accumulators are used to compute a dynamic scaling factor (d₀) that is adjusted during training based on the difference between the weighted squared norm of the accumulated updates and the accumulated squared gradients. The optimizer also supports momentum, decoupled weight decay, and bias correction, and is capable of handling sparse gradients via specialized masking functions.

**Parameters**:

- **`learning_rate`** *(float, default=1.0)*: The base step size for parameter updates.
- **`epsilon`** *(float, default=0.0)*: A small constant for numerical stability, added to denominators to avoid division by zero.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay regularization. When non-zero, weight decay is applied either in a decoupled manner or directly, depending on `weight_decouple`.
- **`momentum`** *(float, default=0.0)*: Momentum factor for smoothing the updates. When set above 0, a momentum update is applied to the parameters.
- **`d0`** *(float, default=1e-6)*: Initial adaptive scaling factor that controls the magnitude of the updates.
- **`growth_rate`** *(float, default=`inf`)*: The maximum factor by which the adaptive scaling factor is allowed to grow.
- **`weight_decouple`** *(bool, default=True)*: Determines whether weight decay is decoupled from the gradient update.
- **`fixed_decay`** *(bool, default=False)*: Uses a fixed weight decay value rather than scaling it by the learning rate.
- **`bias_correction`** *(bool, default=False)*: If enabled, applies bias correction during the computation of the adaptive scaling factor.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by the global norm across all parameters.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average (EMA) to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for updating EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before performing an update.
- **`name`** *(str, default="dadaptadagrad")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from dadaptadagrad import DAdaptAdaGrad

# Instantiate the DAdaptAdaGrad optimizer
optimizer = DAdaptAdaGrad(
    learning_rate=1.0,
    epsilon=0.0,
    weight_decay=1e-2,
    momentum=0.9,
    d0=1e-6,
    growth_rate=float('inf'),
    weight_decouple=True,
    fixed_decay=False,
    bias_correction=False
)

# Compile a model using the DAdaptAdaGrad optimizer
model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

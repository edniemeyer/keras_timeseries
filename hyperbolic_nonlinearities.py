import keras.backend as K
import numpy as np
from keras.engine import Layer
from keras import initializations

def _hyperbolicReLU(x, tau):
    return  (x + K.sqrt(x**2 + tau**2) ) /2

def _assymetricBiHyperbolic(x, lmbda, tau_1, tau_2):
    return K.sqrt(lmbda**2 * (x + 1 / (4*lmbda))**2 + tau_1**2) - K.sqrt(lmbda**2 * (x - 1 / (4*lmbda))**2 + tau_2**2) + 1 / 2

def _ext_assymetricBiHyperbolic_old(x, lmbda, tau_1, tau_2):
    return 2*(K.sqrt(lmbda**2 * (x + 1 / (4*lmbda))**2 + tau_1**2) - K.sqrt(lmbda**2 * (x - 1 / (4*lmbda))**2 + tau_2**2))

def _ext_assymetricBiHyperbolic(x, lmbda, tau_1, tau_2):
    return K.sqrt(lmbda**2 * (x + 1 / (2*lmbda))**2 + tau_1**2) - K.sqrt(lmbda**2 * (x - 1 / (2*lmbda))**2 + tau_2**2)

def _biHyperbolic(x, lmbda, tau):
    return _assymetricBiHyperbolic(x, lmbda, tau, tau)

def _ext_biHyperbolic(x, lmbda, tau):
    return _ext_assymetricBiHyperbolic(x, lmbda, tau, tau)

def _hyperbolic(x, rho):
    return 0.5 * (1 + x/K.sqrt(x**2 + 4*rho**2) )

def _ext_hyperbolic(x, rho):
    return x / K.sqrt(x**2 + rho**2)

class HyperbolicReLU(Layer):

    def __init__(self, tau, **kwargs):
        self.supports_masking = True
        self.tau = K.variable(tau)
        super(HyperbolicReLU, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return _hyperbolicReLU(x, self.tau)

    def get_config(self):
        config = {'tau': self.tau}
        base_config = super(HyperbolicReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AssymetricBiHyperbolic(Layer):

    def __init__(self, lmbda, tau_1, tau_2, **kwargs):
        self.supports_masking = True
        self.lmbda = K.variable(lmbda)
        self.tau_1 = K.variable(tau_1)
        self.tau_2 = K.variable(tau_2)
        super(BiHyperbolic, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return _assymetricBiHyperbolic(x, self.lmbda, self.tau_1, self.tau_2)

    def get_config(self):
        config = {'lmbda': self.lmbda,
                      'tau_1': self.tau_1,
                      'tau_2': self.tau_2}
        base_config = super(BiHyperbolic, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class BiHyperbolic(Layer):

    def __init__(self, lmbda, tau, mode='ext', **kwargs):
        self.supports_masking = True
        self.lmbda = K.variable(lmbda)
        self.tau = K.variable(tau)
        self.mode = mode
        super(BiHyperbolic, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if self.mode == 'basic':
            return _biHyperbolic(x, self.lmbda, self.tau)
        if self.mode == 'ext':
            return _ext_biHyperbolic(x, self.lmbda, self.tau)

    def get_config(self):
        config = {'lmbda': self.lmbda,
                      'tau': self.tau,
                      'mode': self.mode}
        base_config = super(BiHyperbolic, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Hyperbolic(Layer):

    def __init__(self, rho, mode='ext', **kwargs):
        self.supports_masking = True
        self.rho = K.variable(rho)
        self.mode = mode
        super(Hyperbolic, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if self.mode == 'basic':
            return _hyperbolic(x, self.rho)
        else:
            return _ext_hyperbolic(x, self.rho)

    def get_config(self):
        config = {'rho': self.rho, 'mode': self.mode}
        base_config = super(Hyperbolic, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AdaptativeAssymetricBiHyperbolic(Layer):

    def __init__(self, lmbda_init='one', tau_1_init='glorot_normal', tau_2_init='glorot_normal', mode='ext', shared_axes=None, weights=None, **kwargs):
        self.supports_masking = True
        self.lmbda_init = lmbda_init
        self.tau_1_init = tau_1_init
        self.tau_2_init = tau_2_init
        self.mode = mode
        self.initial_weights = weights

        if not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)
        super(AdaptativeAssymetricBiHyperbolic, self).__init__(**kwargs)

    def build(self, input_shape):
        #input_shape = input_shape[1:]
        param_shape = list(input_shape[1:])
        self.param_broadcast = [False] * len(param_shape)
        if self.shared_axes[0] is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
                self.param_broadcast[i - 1] = True
        """
        self.lmbda = K.variable(self.lambda_init * np.ones(input_shape),
                                 name='{}_lambda'.format(self.name))
        self.tau_1 = K.variable(self.tau_1_init * np.ones(input_shape),
                                 name='{}_tau_1'.format(self.name))
        self.tau_2 = K.variable(self.tau_2_init * np.ones(input_shape),
                                 name='{}_tau_2'.format(self.name))
        """
        lmbda_init = initializations.get(self.lmbda_init)
        tau_1_init = initializations.get(self.tau_1_init)
        tau_2_init = initializations.get(self.tau_2_init)

        self.lmbda = lmbda_init(param_shape,
                                    name='{}_lmbda'.format(self.name))
        self.tau_1 = tau_1_init(param_shape,
                                    name='{}_tau_1'.format(self.name))
        self.tau_2 = tau_2_init(param_shape,
                                    name='{}_tau_2'.format(self.name))

        self.trainable_weights = [self.lmbda, self.tau_1, self.tau_2]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights


    def call(self, x, mask=None):
        if self.mode == 'basic':
            return _assymetricBiHyperbolic(x, self.lmbda, self.tau_1, self.tau_2)
        return _ext_assymetricBiHyperbolic(x, self.lmbda, self.tau_1, self.tau_2)

    def get_config(self):
        config = {'lmbda_init': self.lambda_init,
                  'tau_1_init': self.tau_1_init,
                  'tau_2_init': self.tau_2_init}
        base_config = super(AdaptativeAssymetricBiHyperbolic, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AdaptativeBiHyperbolic(Layer):

    def __init__(self, lmbda_init='one', tau_init='glorot_normal', mode='ext', shared_axes=None, weights=None, **kwargs):
        self.supports_masking = True
        self.lmbda_init = lmbda_init
        self.tau_init = tau_init
        self.mode = mode
        self.initial_weights = weights

        if not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)
        super(AdaptativeBiHyperbolic, self).__init__(**kwargs)

    def build(self, input_shape):
        #input_shape = input_shape[1:]
        param_shape = list(input_shape[1:])
        self.param_broadcast = [False] * len(param_shape)
        if self.shared_axes[0] is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
                self.param_broadcast[i - 1] = True

        lmbda_init = initializations.get(self.lmbda_init)
        tau_init = initializations.get(self.tau_init)

        self.lmbda = lmbda_init(param_shape,
                                    name='{}_lmbda'.format(self.name))
        self.tau = tau_init(param_shape,
                                    name='{}_tau'.format(self.name))

        self.trainable_weights = [self.lmbda, self.tau]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights


    def call(self, x, mask=None):
        if self.mode == 'basic':
            return _biHyperbolic(x, self.lmbda, self.tau)
        return _ext_biHyperbolic(x, self.lmbda, self.tau)

    def get_config(self):
        config = {'lmbda_init': self.lambda_init,
                  'tau_init': self.tau_init}
        base_config = super(AdaptativeBiHyperbolic, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AdaptativeHyperbolic(Layer):

    def __init__(self, rho_init='glorot_normal', mode='ext', shared_axes=None, weights=None, **kwargs):
        self.supports_masking = True
        self.rho_init = rho_init
        self.mode = mode
        self.initial_weights = weights

        if not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)
        super(AdaptativeHyperbolic, self).__init__(**kwargs)

    def build(self, input_shape):
        #input_shape = input_shape[1:]
        param_shape = list(input_shape[1:])
        self.param_broadcast = [False] * len(param_shape)
        if self.shared_axes[0] is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
                self.param_broadcast[i - 1] = True

        rho_init = initializations.get(self.rho_init)

        self.rho = rho_init(param_shape,
                                    name='{}_rho'.format(self.name))
        self.trainable_weights = [self.rho]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights


    def call(self, x, mask=None):
        if self.mode == 'basic':
            return _hyperbolic(x, self.rho)
        return _ext_hyperbolic(x, self.rho)

    def get_config(self):
        config = {'rho_init': self.rho_init}
        base_config = super(AdaptativeHyperbolic, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AdaptativeHyperbolicReLU(Layer):
    '''Parametric Hyperbolic Smoothing Rectifier
    '''

    def __init__(self, tau_init='glorot_normal', shared_axes=None, weights=None, **kwargs):
        self.supports_masking = True
        self.tau_init = tau_init
        self.initial_weights = weights

        if not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)
        super(AdaptativeHyperbolicReLU, self).__init__(**kwargs)

    def build(self, input_shape):
        #input_shape = input_shape[1:]
        param_shape = list(input_shape[1:])
        self.param_broadcast = [False] * len(param_shape)
        if self.shared_axes[0] is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
                self.param_broadcast[i - 1] = True

        tau_init = initializations.get(self.tau_init)

        self.tau = tau_init(param_shape,
                                    name='{}_tau'.format(self.name))
        self.trainable_weights = [self.tau]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        return _hyperbolicReLU(x, self.tau)

    def get_config(self):
        config = {'tau_init': self.tau_init}
        base_config = super(AdaptativeHyperbolicReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class PELU(Layer):
    '''Parametric Exponential Linear Unit
    `f(x) = alpha * (exp(x / beta) - 1) for x < 0`,
    `f(x) = alpha / beta * x for x >= 0`.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        alpha_init: float. Initial value of the alpha weights.
        beta_init: float. Initial values of the beta weights.
        weights: initial weights, as a list of 2 numpy arrays.

    # References
        - [Parametric Exponential Linear Unit for Deep Convolutional Neural Networks](https://arxiv.org/abs/1605.09332)
    '''
    def __init__(self, alpha_init=1.0, beta_init=1.0, weights=None, **kwargs):
        self.supports_masking = True
        self.alpha_init = K.cast_to_floatx(alpha_init)
        self.beta_init = K.cast_to_floatx(beta_init)
        self.initial_weights = weights
        super(PELU, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape[1:]
        self.alphas = K.variable(self.alpha_init * np.ones(input_shape),
                                 name='{}_alphas'.format(self.name))
        self.betas = K.variable(self.beta_init * np.ones(input_shape),
                                name='{}_betas'.format(self.name))
        self.trainable_weights = [self.alphas, self.betas]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        pos = K.relu(x) * self.alphas / self.betas
        neg = (x - abs(x)) * 0.5
        neg = self.alphas * (K.exp(neg / self.betas) - 1)
        return pos + neg

    def get_config(self):
        config = {'alpha_init': self.alpha_init,
                  'beta_init': self.beta_init}
        base_config = super(PELU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

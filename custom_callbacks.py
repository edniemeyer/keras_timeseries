from keras.callbacks import Callback
import numpy as np
import time

class ElapsedTime(Callback):

    def __init__(self):
        super(ElapsedTime, self).__init__()
        self.timing = []

    def on_epoch_begin(self, epoch, logs={}):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        elapsed_time = time.time() - self.start_time
        self.timing.append(elapsed_time)

class CriteriaStopping(Callback):
    '''Stop training when a monitored quantity has stopped improving.
    # Arguments
        monitor: quantity to be monitored.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
    '''
    def __init__(self, criteria, monitor='val_acc', verbose=1, mode='auto'):
        super(CriteriaStopping, self).__init__()

        self.monitor = monitor
        self.criteria = criteria
        self.verbose = verbose
        self.wait = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('CriteriaStopping mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Criteria stopping requires %s available!' % (self.monitor), RuntimeWarning)
        if self.monitor_op(current, self.criteria):
            if self.verbose > 0:
                print('Epoch %05d: criteria stopping' % (epoch))
            self.model.stop_training = True

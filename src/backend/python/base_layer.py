from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections as collections_lib
import enum
import functools
import inspect
import re
import numpy as np
from six.moves import zip
import weakref



from backend.python import context
from backend.python.framework import ops
from backend.python.framework import dtypes
from backend.python.ops import variables as tf_variables
from backend.python.training import base as checkpointable
#from backend.util import function_utils
from backend.util import tf_inspect
from backend.util import nest
from backend.python.ops.init_ops import GlorotUniform

PER_GRAPH_LAYER_NAME_UIDS = weakref.WeakKeyDictionary()

class InputSpec(object):
  def __init__(self,
               dtype=None,
               shape=None,
               ndim=None,
               max_ndim=None,
               min_ndim=None,
               axes=None):
    self.dtype = dtype
    self.shape = shape
    if shape is not None:
      self.ndim = len(shape)
    else:
      self.ndim = ndim
    self.max_ndim = max_ndim
    self.min_ndim = min_ndim
    self.axes = axes or {}

class Layer(checkpointable.CheckpointableBase):

  def __init__(self, trainable=True, name=None, dtype=None, **kwargs):
    allowed_kwargs = {
        'input_shape',
        'batch_input_shape',
        'batch_size',
        'weights',
        'activity_regularizer',
    }
    for kwarg in kwargs:
      if kwarg not in allowed_kwargs:
        raise TypeError('Keyword argument not understood:', kwarg)

    self.trainable = trainable
    self.stateful = False
    self.built = False
    self.input_spec = None

    self._init_set_name(name)

    activity_regularizer = kwargs.pop('activity_regularizer', None)
    self._activity_regularizer = activity_regularizer
    self._trainable_weights = []
    self._non_trainable_weights = []
    self._updates = []
    self._callable_losses = []
    self._losses = []
    self._in_call = False 
    self._dtype = None if dtype is None else dtypes.as_dtype(dtype).name
    #self._call_fn_args = function_utils.fn_args(self.call)
    #self._compute_previous_mask = ('mask' in self._call_fn_args or hasattr(self, 'compute_mask'))
    self._call_convention = CallConvention.EXPLICIT_INPUTS_ARGUMENT

    self._inbound_nodes = []
    self._outbound_nodes = []

    self.supports_masking = False


    if 'input_shape' in kwargs or 'batch_input_shape' in kwargs:
      if 'batch_input_shape' in kwargs:
        batch_input_shape = tuple(kwargs['batch_input_shape'])
      elif 'input_shape' in kwargs:
        if 'batch_size' in kwargs:
          batch_size = kwargs['batch_size']
        else:
          batch_size = None
        batch_input_shape = (batch_size,) + tuple(kwargs['input_shape'])
      self._batch_input_shape = batch_input_shape

    if 'weights' in kwargs:
      self._initial_weights = kwargs['weights']
    else:
      self._initial_weights = None

  def _init_set_name(self, name, zero_based=True):
    if not name:
      self._name = unique_layer_name(
          to_snake_case(self.__class__.__name__),
          zero_based=zero_based)
    else:
      self._name = name

  @property
  def dtype(self):
    return self._dtype

  @property
  def name(self):
    return self._name

  @property
  def activity_regularizer(self):
    return self._activity_regularizer

  @activity_regularizer.setter
  def activity_regularizer(self, regularizer):
    self._activity_regularizer = self._no_dependency(regularizer)

  @property
  def trainable_weights(self):
    return self._trainable_weights if self.trainable else []

  @property
  def non_trainable_weights(self):
    if self.trainable:
      return self._non_trainable_weights
    else:
      return self._trainable_weights + self._non_trainable_weights

  @property
  def trainable_variables(self):
    return self.trainable_weights

  @property
  def non_trainable_variables(self):
    return self.non_trainable_weights

  @property
  def weights(self):
    return self.trainable_weights + self.non_trainable_weights

  def add_update(self, updates, inputs=None):
    if context.executing_eagerly():
      return

    def process_update(x):
      if isinstance(x, ops.Operation):
        return x
      elif hasattr(x, 'op'):
        return x.op
      else:
        return ops.convert_to_tensor(x)

    updates = to_list(updates)
    updates = [process_update(x) for x in updates]
    self._updates += updates
    if inputs is None:
      for u in updates:
        u._unconditional_update = True  
    else:
      for u in updates:
        u._unconditional_update = False  

  def add_variable(self, *args, **kwargs):
    return self.add_weight(*args, **kwargs)

  def add_weight(self,
                 name,
                 shape,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=None,
                 constraint=None,
                 partitioner=None,
                 use_resource=None,
                 synchronization=tf_variables.VariableSynchronization.AUTO,
                 aggregation=tf_variables.VariableAggregation.NONE,
                 **kwargs):
    
    for kwarg in kwargs:
      if kwarg not in ['getter', 'collections']:
        raise TypeError('Unknown keyword argument:', kwarg)
    getter = kwargs.pop('getter', None)
    collections = kwargs.pop('collections', None)

    if dtype is None:
      dtype = self.dtype or _FLOATX
    dtype = dtypes.as_dtype(dtype)

    constraint = get(constraint)

    if synchronization == tf_variables.VariableSynchronization.ON_READ:
      if trainable:
        raise ValueError(
            'Synchronization value can be set to '
            'VariableSynchronization.ON_READ only for non-trainable variables. '
            'You have specified trainable=True and '
            'synchronization=VariableSynchronization.ON_READ.')
      else:
        trainable = False
    elif trainable is None:
      trainable = True

    if initializer is None:
      if dtype.is_floating:
        initializer = GlorotUniform()
      elif dtype.is_integer or dtype.is_unsigned or dtype.is_bool:
        initializer = initializers.zeros()
      else:
        raise ValueError('An initializer for variable %s of type %s is required'
                         ' for layer %s' % (name, dtype.base_dtype, self.name))

    variable = self._add_variable_with_custom_getter(
        name=name,
        shape=shape,
        getter=getter or make_variable,
        overwrite=True,
        initializer=initializer,
        dtype=dtype,
        constraint=constraint,
        trainable=trainable and self.trainable,
        partitioner=partitioner,
        use_resource=use_resource,
        collections=collections,
        synchronization=synchronization,
        aggregation=aggregation)


    if regularizer is not None:
      self._handle_weight_regularization(name, variable, regularizer)

    if trainable:
      self._trainable_weights.append(variable)
    else:
      self._non_trainable_weights.append(variable)
    return variable

  def _handle_activity_regularization(self, inputs, outputs):
    if self._activity_regularizer:
      output_list = nest.flatten(outputs)
      for output in output_list:
        with ops.name_scope('ActivityRegularizer'):
          activity_regularization = self._activity_regularizer(output)
        self.add_loss(activity_regularization, inputs=inputs)

  def __call__(self, inputs, *args, **kwargs):
    input_list = nest.flatten(inputs)

    build_graph = not context.executing_eagerly()
    in_deferred_mode = isinstance(input_list[0], DeferredTensor)

    previous_mask = None
    

    input_shapes = None

    with ops.name_scope(self._name_scope()):
      if not self.built:
        if not build_graph:
          if self._activity_regularizer:
            raise ValueError(
                'activity_regularizer currently unsupported with '
                'eager execution enabled. Found an activity_regularizer in '
                '%s(%s).' % (self.__class__.__name__, self))
        if not build_graph and not in_deferred_mode:
          for x in input_list:
            if hasattr(x, '_keras_history'):
              raise ValueError('_keras_history currently unsupported in '
                               'Eager mode. Found _keras_history in %s while '
                               'executing __call__ for %s(%s)' %
                               (x, self.__class_.__name__, self))

        self._assert_input_compatibility(inputs)
        if input_list and self._dtype is None:
          try:
            self._dtype = input_list[0].dtype.base_dtype.name
          except AttributeError:
            pass

        if all(hasattr(x, 'shape') for x in input_list):
          input_shapes = nest.map_structure(lambda x: x.shape, inputs)

        if (not hasattr(self, '_is_graph_network') or
            self.__class__.__name__ == 'Sequential' or
            not hasattr(self.build, '_is_default')):
          self.build(input_shapes)
        self.built = True

      if build_graph or in_deferred_mode:
        self._assert_input_compatibility(inputs)

      if not in_deferred_mode:
        self._in_call = True
        outputs = self.call(inputs, *args, **kwargs)
        self._in_call = False
        if outputs is None:
          raise ValueError('A layer\'s `call` method should return a Tensor '
                           'or a list of Tensors, not None (layer: ' +
                           self.name + ').')
      else:
        if input_shapes is None:
          input_shapes = nest.map_structure(lambda x: x.shape, inputs)

        output_shapes = self.compute_output_shape(input_shapes)
        output_shapes = nest.flatten(output_shapes)
        outputs = [
            DeferredTensor(shape=shape, dtype=self._dtype)
            for shape in output_shapes
        ]
        if len(outputs) == 1:
          outputs = outputs[0]

      if build_graph:
        self._handle_activity_regularization(inputs, outputs)
        self._set_mask_metadata(inputs, outputs, previous_mask)

      if in_deferred_mode or build_graph and have_all_keras_metadata(inputs):
        inputs, outputs = self._set_connectivity_metadata_(
            inputs, outputs, args, kwargs)
      if context.executing_eagerly():
        return outputs

      if hasattr(self, '_symbolic_set_inputs') and not self.inputs:
        self._symbolic_set_inputs(inputs, outputs)

      if in_deferred_mode or build_graph:
        self._set_learning_phase_metadata(inputs, outputs)


    if hasattr(self, '_initial_weights') and self._initial_weights is not None:
      self.set_weights(self._initial_weights)
      del self._initial_weights
    return outputs

  def apply(self, inputs, *args, **kwargs):
    return self.__call__(inputs, *args, **kwargs)

  def _set_learning_phase_metadata(self, inputs, outputs):
    output_tensors = to_list(outputs)
    uses_lp = any(
        [getattr(x, '_uses_learning_phase', False)
         for x in to_list(inputs)])
    uses_lp = getattr(self, 'uses_learning_phase', False) or uses_lp
    for i in range(len(output_tensors)):
      try:
        output_tensors[i]._uses_learning_phase = getattr(
            output_tensors[i], '_uses_learning_phase', False) or uses_lp
      except AttributeError:
        pass

  def _set_mask_metadata(self, inputs, outputs, previous_mask):
    mask_already_computed = all(
        hasattr(x, '_keras_mask') for x in to_list(outputs))
    if hasattr(self, 'compute_mask') and not mask_already_computed:
      output_mask = self.compute_mask(inputs, previous_mask)
    else:
      output_mask = None
    if isinstance(outputs, (list, tuple)):
      if output_mask is None:
        output_mask = [None for _ in range(len(outputs))]
      for x, m in zip(outputs, output_mask):
        try:
          x._keras_mask = m  
        except AttributeError:
          pass 
    else:
      try:
        outputs._keras_mask = output_mask  
      except AttributeError:
        pass 

  def _assert_input_compatibility(self, inputs):
    if not self.input_spec:
      return
    if not isinstance(self.input_spec, (list, tuple)):
      input_spec = nest.flatten(self.input_spec)
    else:
      input_spec = self.input_spec
    inputs = nest.flatten(inputs)
    if len(inputs) != len(input_spec):
      raise ValueError('Layer ' + self.name + ' expects ' +
                       str(len(input_spec)) + ' inputs, '
                       'but it received ' + str(len(inputs)) +
                       ' input tensors. Inputs received: ' + str(inputs))
    for input_index, (x, spec) in enumerate(zip(inputs, input_spec)):
      if spec is None:
        continue

      if (spec.ndim is not None or
          spec.min_ndim is not None or
          spec.max_ndim is not None):
        if x.shape.ndims is None:
          raise ValueError('Input ' + str(input_index) + ' of layer ' +
                           self.name + ' is incompatible with the layer: '
                           'its rank is undefined, but the layer requires a '
                           'defined rank.')

      if spec.ndim is not None:
        ndim = x.shape.ndims
        if ndim != spec.ndim:
          raise ValueError('Input ' + str(input_index) + ' of layer ' +
                           self.name + ' is incompatible with the layer: '
                           'expected ndim=' + str(spec.ndim) + ', found ndim=' +
                           str(ndim) + '. Full shape received: ' +
                           str(x.shape.as_list()))
      if spec.max_ndim is not None:
        ndim = x.shape.ndims
        if ndim is not None and ndim > spec.max_ndim:
          raise ValueError('Input ' + str(input_index) + ' of layer ' +
                           self.name + ' is incompatible with the layer: '
                           'expected max_ndim=' + str(spec.max_ndim) +
                           ', found ndim=' + str(ndim))
      if spec.min_ndim is not None:
        ndim = x.shape.ndims
        if ndim is not None and ndim < spec.min_ndim:
          raise ValueError('Input ' + str(input_index) + ' of layer ' +
                           self.name + ' is incompatible with the layer: '
                           ': expected min_ndim=' + str(spec.min_ndim) +
                           ', found ndim=' + str(ndim) +
                           '. Full shape received: ' +
                           str(x.shape.as_list()))
      if spec.dtype is not None:
        if x.dtype != spec.dtype:
          raise ValueError('Input ' + str(input_index) + ' of layer ' +
                           self.name + ' is incompatible with the layer: '
                           'expected dtype=' + str(spec.dtype) +
                           ', found dtype=' + str(x.dtype))
      if spec.axes:
        shape = x.shape.as_list()
        if shape is not None:
          for axis, value in spec.axes.items():
            if hasattr(value, 'value'):
              value = value.value
            if value is not None and shape[int(axis)] not in {value, None}:
              raise ValueError(
                  'Input ' + str(input_index) + ' of layer ' + self.name + ' is'
                  ' incompatible with the layer: expected axis ' + str(axis) +
                  ' of input shape to have value ' + str(value) +
                  ' but received input with shape ' + str(shape))
      if spec.shape is not None:
        shape = x.shape.as_list()
        if shape is not None:
          for spec_dim, dim in zip(spec.shape, shape):
            if spec_dim is not None and dim is not None:
              if spec_dim != dim:
                raise ValueError('Input ' + str(input_index) +
                                 ' is incompatible with layer ' + self.name +
                                 ': expected shape=' + str(spec.shape) +
                                 ', found shape=' + str(shape))

def to_snake_case(name):
  intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
  insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
  if insecure[0] != '_':
    return insecure
  return 'private' + insecure

def unique_layer_name(name, name_uid_map=None, avoid_names=None, namespace='',
                      zero_based=False):
  if name_uid_map is None:
    name_uid_map = get_default_graph_uid_map()
  if avoid_names is None:
    avoid_names = set()
  proposed_name = None
  while proposed_name is None or proposed_name in avoid_names:
    name_key = (namespace, name)
    if zero_based:
      number = name_uid_map[name_key]
      if number:
        proposed_name = name + '_' + str(number)
      else:
        proposed_name = name
      name_uid_map[name_key] += 1
    else:
      name_uid_map[name_key] += 1
      proposed_name = name + '_' + str(name_uid_map[name_key])
  return proposed_name

def get_default_graph_uid_map():
  graph = ops.get_default_graph()
  name_uid_map = PER_GRAPH_LAYER_NAME_UIDS.get(graph, None)
  if name_uid_map is None:
    name_uid_map = collections_lib.defaultdict(int)
    PER_GRAPH_LAYER_NAME_UIDS[graph] = name_uid_map
  return name_uid_map

class CallConvention(enum.Enum):
  EXPLICIT_INPUTS_ARGUMENT = 1
  SINGLE_POSITIONAL_ARGUMENT = 2
  POSITIONAL_ARGUMENTS_ARE_INPUTS = 3

class DeferredTensor(object):
  def __init__(self, shape, dtype, name=None):
    self.shape = tensor_shape.TensorShape(shape)
    if dtype is None:
      self.dtype = dtypes.as_dtype(np.float32)
    else:
      self.dtype = dtypes.as_dtype(dtype)
    self.name = name

  def get_shape(self):
    return self.shape

  def __str__(self):
    return "DeferredTensor('%s', shape=%s, dtype=%s)" % (self.name,
                                                         self.shape,
                                                         self.dtype.name)

  def __repr__(self):
    return "<DeferredTensor '%s' shape=%s dtype=%s>" % (self.name,
                                                        self.shape,
                                                        self.dtype.name)

def get(identifier):
  if identifier is None:
    return None
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, six.string_types):
    config = {'class_name': str(identifier), 'config': {}}
    return deserialize(config)
  elif callable(identifier):
    return identifier
  else:
    raise ValueError('Could not interpret constraint identifier: ' +
                     str(identifier))

def to_list(x):
  if isinstance(x, list):
    return x
  return [x]

def have_all_keras_metadata(iterable_or_element):
  if not isinstance(iterable_or_element, (list, tuple)):
    iterable = [iterable_or_element]
  else:
    iterable = iterable_or_element
  return all([hasattr(x, '_keras_history') for x in iterable])

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections as collections_lib
import enum  # pylint: disable=g-bad-import-order
import functools
import inspect  # Necessary supplement to tf_inspect to deal with variadic args.
import re
import numpy as np
from six.moves import zip  # pylint: disable=redefined-builtin


from tensorflow.python import function as eager_function
from tensorflow.python import context
from tensorflow.python.keras import constraints
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import initializers
from tensorflow.python.keras import backend
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training import base as checkpointable
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import nest
from tensorflow import doc_controls


@tf_export('keras.layers.InputSpec', 'layers.InputSpec')
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

  def __repr__(self):
    spec = [('dtype=' + str(self.dtype)) if self.dtype else '',
            ('shape=' + str(self.shape)) if self.shape else '',
            ('ndim=' + str(self.ndim)) if self.ndim else '',
            ('max_ndim=' + str(self.max_ndim)) if self.max_ndim else '',
            ('min_ndim=' + str(self.min_ndim)) if self.min_ndim else '',
            ('axes=' + str(self.axes)) if self.axes else '']
    return 'InputSpec(%s)' % ', '.join(x for x in spec if x)


@tf_export('keras.layers.Layer')
class Layer(checkpointable.CheckpointableBase):
  @checkpointable.no_automatic_dependency_tracking
  def __init__(self, trainable=True, name=None, dtype=None, **kwargs):
    # These properties should be set by the user via keyword arguments.
    # note that 'dtype', 'input_shape' and 'batch_input_shape'
    # are only applicable to input layers: do not pass these keywords
    # to non-input layers.
    allowed_kwargs = {
        'input_shape',
        'batch_input_shape',
        'batch_size',
        'weights',
        'activity_regularizer',
    }
    # Validate optional keyword arguments.
    for kwarg in kwargs:
      if kwarg not in allowed_kwargs:
        raise TypeError('Keyword argument not understood:', kwarg)

    # Mutable properties
    # Indicates whether the layer's weights are updated during training
    # and whether the layer's updates are run during training
    self.trainable = trainable
    # A stateful layer is a layer whose updates are run during inference too,
    # for instance stateful RNNs.
    self.stateful = False
    # Indicates whether `build` needs to be called upon layer call, to create
    # the layer's weights.
    self.built = False
    # Provides information about which inputs are compatible with the layer.
    self.input_spec = None

    self._init_set_name(name)

    activity_regularizer = kwargs.pop('activity_regularizer', None)
    if activity_regularizer and context.executing_eagerly():
      raise ValueError(
          ('Activity regularization is not supported when executing eagerly. '
           'Got activity_regularizer=%s') % (activity_regularizer,))
    self._activity_regularizer = activity_regularizer
    self._trainable_weights = []
    self._non_trainable_weights = []
    self._updates = []
    # A list of zero-argument lambdas which return Tensors, used for variable
    # regularizers.
    self._callable_losses = []
    # A list of Tensors containing activity regularizers and losses manually
    # added through `add_loss`. Empty when executing eagerly.
    self._losses = []
    self._in_call = False  # Flag for error checking in add_loss
    self._dtype = None if dtype is None else dtypes.as_dtype(dtype).name
    self._call_fn_args = function_utils.fn_args(self.call)
    self._compute_previous_mask = ('mask' in self._call_fn_args or
                                   hasattr(self, 'compute_mask'))
    self._call_convention = CallConvention.EXPLICIT_INPUTS_ARGUMENT

    # These lists will be filled via successive calls
    # to self._add_inbound_node().
    self._inbound_nodes = []
    self._outbound_nodes = []

    self.supports_masking = False

    call_argspec = tf_inspect.getfullargspec(self.call)
    if 'training' in call_argspec.args:
      self._expects_training_arg = True
    else:
      self._expects_training_arg = False

    # Manage input shape information if passed.
    if 'input_shape' in kwargs or 'batch_input_shape' in kwargs:
      # In this case we will later create an input layer
      # to insert before the current layer
      if 'batch_input_shape' in kwargs:
        batch_input_shape = tuple(kwargs['batch_input_shape'])
      elif 'input_shape' in kwargs:
        if 'batch_size' in kwargs:
          batch_size = kwargs['batch_size']
        else:
          batch_size = None
        batch_input_shape = (batch_size,) + tuple(kwargs['input_shape'])
      self._batch_input_shape = batch_input_shape

    # Manage initial weight values if passed.
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

  @property
  def variables(self):
    return self.weights

  @property
  def updates(self):
    if context.executing_eagerly():
      raise RuntimeError('Layer.updates not supported in Eager mode.')
    if not self.trainable and not self.stateful:
      return []
    return self._updates

  @doc_controls.for_subclass_implementers
  def add_update(self, updates, inputs=None):
    if context.executing_eagerly():
      return  # Updates already applied when in eager mode.

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
        u._unconditional_update = True  # pylint: disable=protected-access
    else:
      for u in updates:
        u._unconditional_update = False  # pylint: disable=protected-access

  def get_updates_for(self, inputs):
    if context.executing_eagerly():
      raise RuntimeError('`get_updates_for()` not supported in Eager mode.')

    # Updates disabled if layer is not trainable and not explicitly stateful.
    if not self.trainable and not self.stateful:
      return []

    if inputs is None:
      # Requesting unconditional updates.
      return [x for x in self.updates if x._unconditional_update]  # pylint: disable=protected-access

    # Requesting input-conditional updates.
    inputs = nest.flatten(inputs)
    reachable = tf_utils.get_reachable_from_inputs(inputs, self.updates)
    updates = []
    for update in self.updates:
      if update in reachable:
        updates.append(update)
    return updates

  @property
  def losses(self):
    collected_losses = []
    collected_losses.extend(self._losses)
    for regularizer in self._callable_losses:
      loss_tensor = regularizer()
      if loss_tensor is not None:
        collected_losses.append(loss_tensor)
    return collected_losses

  @doc_controls.for_subclass_implementers
  def add_loss(self, losses, inputs=None):
    executing_eagerly = context.executing_eagerly()
    if executing_eagerly:
      if inputs is not None:
        raise RuntimeError(
            'Activity regularization (via the "inputs" argument to '
            'Layer.add_loss) is not supported when executing eagerly. Consider '
            'returning activity regularization losses from a Model\'s call() '
            'method.')
      if getattr(self, '_in_call', False):
        # TODO(psv): Support activity regularization and a way to reset losses.
        raise RuntimeError(
            'Adding losses inside a Layer\'s call() method is not currently '
            'supported when executing eagerly. Please file a feature request '
            'if you need this limitation lifted.')
    losses = to_list(losses)

    def _tag_unconditional(loss):
      if callable(loss):
        loss = loss()
      if loss is None:
        return None  # Will be filtered out when computing the .losses property
      if not tensor_util.is_tensor(loss):
        loss = ops.convert_to_tensor(loss, dtype=_FLOATX)
      loss._unconditional_loss = (inputs is None)  # pylint: disable=protected-access
      return loss

    for loss in losses:
      if callable(loss):
        self._callable_losses.append(
            functools.partial(_tag_unconditional, loss))
      else:
        if executing_eagerly:
          raise RuntimeError(
              'Layer.add_loss only supported for zero-argument lambdas when '
              'executing eagerly.')
        self._losses.append(_tag_unconditional(loss))

  def get_losses_for(self, inputs):
    if context.executing_eagerly():
      raise RuntimeError('Layer.get_losses_for not supported in Eager mode.')

    if inputs is None:
      # Requesting unconditional losses.
      return [x for x in self.losses if x._unconditional_loss]  # pylint: disable=protected-access

    # Requesting input-conditional losses.
    inputs = nest.flatten(inputs)
    # Retrieve the set of tensors in the TF graph that depend on `inputs`.
    # The losses we want to return will be part of this set.
    # To avoid unnecessary work, we stop the search in case all of
    # `self.losses` have been retrieved.
    reachable = tf_utils.get_reachable_from_inputs(inputs, self.losses)
    losses = []
    for loss in self.losses:
      if loss in reachable:
        losses.append(loss)
    return losses

  def _name_scope(self):
    return self.name

  def build(self, input_shape):
    self.built = True

  @doc_controls.for_subclass_implementers
  def add_variable(self, *args, **kwargs):
    return self.add_weight(*args, **kwargs)

  @doc_controls.for_subclass_implementers
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
    
    # Validate optional keyword arguments.
    for kwarg in kwargs:
      if kwarg not in ['getter', 'collections']:
        raise TypeError('Unknown keyword argument:', kwarg)
    getter = kwargs.pop('getter', None)
    collections = kwargs.pop('collections', None)

    if dtype is None:
      dtype = self.dtype or _FLOATX
    dtype = dtypes.as_dtype(dtype)
    initializer = initializers.get(initializer)
    regularizer = regularizers.get(regularizer)
    constraint = constraints.get(constraint)

    if synchronization == tf_variables.VariableSynchronization.ON_READ:
      if trainable:
        raise ValueError(
            'Synchronization value can be set to '
            'VariableSynchronization.ON_READ only for non-trainable variables. '
            'You have specified trainable=True and '
            'synchronization=VariableSynchronization.ON_READ.')
      else:
        # Set trainable to be false when variable is to be synced on read.
        trainable = False
    elif trainable is None:
      trainable = True

    # Initialize variable when no initializer provided
    if initializer is None:
      # If dtype is DT_FLOAT, provide a uniform unit scaling initializer
      if dtype.is_floating:
        initializer = initializers.glorot_uniform()
      # If dtype is DT_INT/DT_UINT, provide a default value `zero`
      # If dtype is DT_BOOL, provide a default value `FALSE`
      elif dtype.is_integer or dtype.is_unsigned or dtype.is_bool:
        initializer = initializers.zeros()
      # NOTES:Do we need to support for handling DT_STRING and DT_COMPLEX here?
      else:
        raise ValueError('An initializer for variable %s of type %s is required'
                         ' for layer %s' % (name, dtype.base_dtype, self.name))

    variable = self._add_variable_with_custom_getter(
        name=name,
        shape=shape,
        # TODO(allenl): a `make_variable` equivalent should be added as a
        # `Checkpointable` method.
        getter=getter or make_variable,
        # Manage errors in Layer rather than Checkpointable.
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
    backend.track_variable(variable)

    if regularizer is not None:
      # TODO(fchollet): in the future, this should be handled at the
      # level of variable creation, and weight regularization losses
      # should be variable attributes.
      self._handle_weight_regularization(name, variable, regularizer)

    if trainable:
      self._trainable_weights.append(variable)
    else:
      self._non_trainable_weights.append(variable)
    return variable

  def _handle_weight_regularization(self, name, variable, regularizer):
    def _loss_for_variable(v):
      with ops.colocate_with(v):
        with ops.name_scope(name + '/Regularizer'):
          regularization = regularizer(v)
      return regularization

    if isinstance(variable, tf_variables.PartitionedVariable):
      for v in variable:
        self.add_loss(functools.partial(_loss_for_variable, v))
    else:
      self.add_loss(functools.partial(_loss_for_variable, variable))

  def _handle_activity_regularization(self, inputs, outputs):
    # Apply activity regularization.
    # Note that it should be applied every time the layer creates a new
    # output, since it is output-specific.
    if self._activity_regularizer:
      output_list = nest.flatten(outputs)
      for output in output_list:
        with ops.name_scope('ActivityRegularizer'):
          activity_regularization = self._activity_regularizer(output)
        self.add_loss(activity_regularization, inputs=inputs)

  @doc_controls.for_subclass_implementers
  def call(self, inputs, **kwargs):  # pylint: disable=unused-argument
    return inputs

  def __call__(self, inputs, *args, **kwargs):
    input_list = nest.flatten(inputs)

    build_graph = not context.executing_eagerly()
    # TODO(fchollet, allenl): Make deferred mode work with subclassed Models
    # which don't use an "inputs" argument.
    in_deferred_mode = isinstance(input_list[0], DeferredTensor)

    # Handle Keras mask propagation from previous layer to current layer.
    previous_mask = None
    if build_graph and (not hasattr(self, '_compute_previous_mask') or
                        self._compute_previous_mask):
      previous_mask = collect_previous_mask(inputs)
      if not hasattr(self, '_call_fn_args'):
        self._call_fn_args = self._no_dependency(
            function_utils.fn_args(self.call))
      if ('mask' in self._call_fn_args and 'mask' not in kwargs and
          not is_all_none(previous_mask)):
        # The previous layer generated a mask, and mask was not explicitly pass
        # to __call__, hence we set previous_mask as the default value.
        kwargs['mask'] = previous_mask

    input_shapes = None

    with ops.name_scope(self._name_scope()):
      if not self.built:
        if not build_graph:
          # Activity regularization is currently unsupported in Eager mode.
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

        # Check input assumptions set before layer building, e.g. input rank.
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
          # Only if self is a layer, an instance of a sequential model, or
          # the user has manually overwritten the build method do we need to
          # build it.
          self.build(input_shapes)
        # We must set self.built since user defined build functions are not
        # constrained to set self.built.
        self.built = True

      # Check input assumptions set after layer building, e.g. input shape.
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
        # Deferred mode behavior: use `compute_output_shape` to
        # infer the number of outputs of the layer and their shapes.
        if input_shapes is None:
          input_shapes = nest.map_structure(lambda x: x.shape, inputs)

        output_shapes = self.compute_output_shape(input_shapes)
        output_shapes = nest.flatten(output_shapes)
        outputs = [
            # TODO(fchollet): name the deferred tensors?
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
        # Subclassed network: explicitly set metadata normally set by a call to
        # self._set_inputs(). This is not relevant in eager execution.
        self._symbolic_set_inputs(inputs, outputs)

      if in_deferred_mode or build_graph:
        self._set_learning_phase_metadata(inputs, outputs)

    # Optionally load weight values that were specified at layer instantiation.
    # TODO(fchollet): consider enabling this with eager execution too.
    if hasattr(self, '_initial_weights') and self._initial_weights is not None:
      self.set_weights(self._initial_weights)
      del self._initial_weights
    return outputs

  def apply(self, inputs, *args, **kwargs):
    return self.__call__(inputs, *args, **kwargs)

  def _set_learning_phase_metadata(self, inputs, outputs):
    # Update learning phase info. To work with subclassed models,
    # this should be done even if Keras metadata is absent.
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
        # An output element happens to be a C type (such as tuple or dict).
        # We don't track learning phase info in such edge cases.
        pass

  def _set_mask_metadata(self, inputs, outputs, previous_mask):
    # In some cases the mask of the outputs has already been computed by
    # inner layers and does not need to be recomputed by this layer.
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
          x._keras_mask = m  # pylint: disable=protected-access
        except AttributeError:
          pass  # C type such as dict. Masking not supported in this case.
    else:
      try:
        outputs._keras_mask = output_mask  # pylint: disable=protected-access
      except AttributeError:
        pass  # C type such as dict. Masking not supported in this case.

  def _set_connectivity_metadata_(self, inputs, outputs, args, kwargs):
    call_convention = getattr(self, '_call_convention',
                              CallConvention.EXPLICIT_INPUTS_ARGUMENT)
    if args:
      if call_convention == CallConvention.EXPLICIT_INPUTS_ARGUMENT:
        raise TypeError(
            'This Layer takes an `inputs` argument to call(), and only the '
            '`inputs` argument may be specified as a positional argument. '
            'Pass everything else as a keyword argument (those arguments will'
            ' not be tracked as inputs to the Layer).')
      elif call_convention == CallConvention.SINGLE_POSITIONAL_ARGUMENT:
        raise TypeError(
            'This Layer takes a single positional argument to call(), which is '
            'by convention the inputs argument, and only this argument may be '
            'specified as a positional argument. Pass everything else as a '
            'keyword argument (those arguments will not be tracked as inputs '
            'to the Layer).')

    # If the layer returns tensors from its inputs, unmodified,
    # we copy them to avoid loss of tensor metadata.
    output_ls = nest.flatten(outputs)
    output_ls_copy = []
    for x in output_ls:
      if x in nest.flatten(inputs):
        with ops.name_scope(self.name):
          x = array_ops.identity(x)
      output_ls_copy.append(x)
    if len(output_ls_copy) == 1:
      outputs = output_ls_copy[0]
    else:
      outputs = output_ls_copy

    inputs, kwargs = self._inputs_from_call_args(
        call_args=(inputs,) + args, call_kwargs=kwargs)
    # Add an inbound node to the layer, so it can keep track of this call.
    # This updates the layer history of the output tensor(s).
    kwargs.pop('mask', None)  # `mask` should not be serialized.
    self._add_inbound_node(
        input_tensors=inputs, output_tensors=outputs, arguments=kwargs)
    return inputs, outputs

  def _inputs_from_call_args(self, call_args, call_kwargs):
    call_convention = getattr(self, '_call_convention',
                              CallConvention.EXPLICIT_INPUTS_ARGUMENT)
    if (call_convention in (
        CallConvention.EXPLICIT_INPUTS_ARGUMENT,
        CallConvention.SINGLE_POSITIONAL_ARGUMENT)):
      assert len(call_args) == 1  # TypeError raised earlier in __call__.
      return call_args[0], call_kwargs
    else:
      call_arg_spec = tf_inspect.getfullargspec(self.call)
      # There is no explicit "inputs" argument expected or provided to
      # call(). Arguments which have default values are considered non-inputs,
      # and arguments without are considered inputs.
      if call_arg_spec.defaults:
        if call_arg_spec.varargs is not None:
          raise TypeError(
              'Layer.call() may not accept both *args and arguments with '
              'default values (unable to determine which are inputs to the '
              'Layer).')
        keyword_arg_names = set(
            call_arg_spec.args[-len(call_arg_spec.defaults):])
      else:
        keyword_arg_names = set()
        # Training is never an input argument name, to allow signatures like
        # call(x, training).
      keyword_arg_names.add('training')
      _, unwrapped_call = tf_decorator.unwrap(self.call)
      bound_args = inspect.getcallargs(
          unwrapped_call, *call_args, **call_kwargs)
      if call_arg_spec.varkw is not None:
        var_kwargs = bound_args.pop(call_arg_spec.varkw)
        bound_args.update(var_kwargs)
        keyword_arg_names = keyword_arg_names.union(var_kwargs.keys())
      all_args = call_arg_spec.args
      if all_args and bound_args[all_args[0]] is self:
        # Ignore the 'self' argument of methods
        bound_args.pop(call_arg_spec.args[0])
        all_args = all_args[1:]
      non_input_arg_values = {}
      input_arg_values = []
      remaining_args_are_keyword = False
      for argument_name in all_args:
        if argument_name in keyword_arg_names:
          remaining_args_are_keyword = True
        else:
          if remaining_args_are_keyword:
            raise TypeError(
                'Found a positional argument to call() after a non-input '
                'argument. All arguments after "training" must be keyword '
                'arguments, and are not tracked as inputs to the Layer.')
        if remaining_args_are_keyword:
          non_input_arg_values[argument_name] = bound_args[argument_name]
        else:
          input_arg_values.append(bound_args[argument_name])
      if call_arg_spec.varargs is not None:
        input_arg_values.extend(bound_args[call_arg_spec.varargs])
      return input_arg_values, non_input_arg_values

  def compute_output_shape(self, input_shape):
    if context.executing_eagerly():
      self.build(input_shape)

      with context.graph_mode():
        graph = eager_function.FuncGraph('graph')
        with graph.as_default():
          if isinstance(input_shape, list):
            inputs = [generate_placeholders_from_shape(shape)
                      for shape in input_shape]
          else:
            inputs = generate_placeholders_from_shape(input_shape)

          try:
            if self._expects_training_arg:
              outputs = self(inputs, training=False)
            else:
              outputs = self(inputs)
          except TypeError:
            raise NotImplementedError('We could not automatically infer '
                                      'the static shape of the layer\'s output.'
                                      ' Please implement the '
                                      '`compute_output_shape` method on your '
                                      'layer (%s).' % self.__class__.__name__)
      if isinstance(outputs, list):
        return [output.shape for output in outputs]
      else:
        return outputs.shape
    raise NotImplementedError

  def compute_mask(self, inputs, mask=None):  # pylint: disable=unused-argument
    if not self.supports_masking:
      if mask is not None:
        if isinstance(mask, list):
          if any(m is not None for m in mask):
            raise TypeError('Layer ' + self.name + ' does not support masking, '
                            'but was passed an input_mask: ' + str(mask))
        else:
          raise TypeError('Layer ' + self.name + ' does not support masking, '
                          'but was passed an input_mask: ' + str(mask))
      # masking not explicitly supported: return None as mask
      return None
    # if masking is explicitly supported, by default
    # carry over the input mask
    return mask

  def _add_inbound_node(self,
                        input_tensors,
                        output_tensors,
                        arguments=None):
    input_tensors = nest.flatten(input_tensors)
    output_tensors = nest.flatten(output_tensors)

    # Collect input tensor(s) coordinates.
    inbound_layers = []
    node_indices = []
    tensor_indices = []
    for x in input_tensors:
      assert hasattr(x, '_keras_history')
      inbound_layer, node_index, tensor_index = x._keras_history  # pylint: disable=protected-access
      inbound_layers.append(inbound_layer)
      node_indices.append(node_index)
      tensor_indices.append(tensor_index)

    # Create node, add it to inbound nodes.
    Node(
        self,
        inbound_layers=inbound_layers,
        node_indices=node_indices,
        tensor_indices=tensor_indices,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        arguments=arguments)

    # Update tensor history metadata.
    for i in range(len(output_tensors)):
      # The metadata attribute consists of 1) a layer instance
      # 2) a node index for the layer, 3) a tensor index for the node.
      # The allows layer reuse (multiple nodes per layer) and multi-output
      # or multi-input layers (e.g. a layer can return multiple tensors,
      # and each can be sent to a different layer).
      output_tensors[i]._keras_history = (self, len(self._inbound_nodes) - 1, i)  # pylint: disable=protected-access

  def _get_node_attribute_at_index(self, node_index, attr, attr_name):
    if not self._inbound_nodes:
      raise RuntimeError('The layer has never been called '
                         'and thus has no defined ' + attr_name + '.')
    if not len(self._inbound_nodes) > node_index:
      raise ValueError('Asked to get ' + attr_name + ' at node ' +
                       str(node_index) + ', but the layer has only ' +
                       str(len(self._inbound_nodes)) + ' inbound nodes.')
    values = getattr(self._inbound_nodes[node_index], attr)
    if len(values) == 1:
      return values[0]
    else:
      return values

  def get_input_mask_at(self, node_index):
    inputs = self.get_input_at(node_index)
    if isinstance(inputs, list):
      return [getattr(x, '_keras_mask', None) for x in inputs]
    else:
      return getattr(inputs, '_keras_mask', None)

  def get_output_mask_at(self, node_index):
    output = self.get_output_at(node_index)
    if isinstance(output, list):
      return [getattr(x, '_keras_mask', None) for x in output]
    else:
      return getattr(output, '_keras_mask', None)

  @property
  def input_mask(self):
    inputs = self.input
    if isinstance(inputs, list):
      return [getattr(x, '_keras_mask', None) for x in inputs]
    else:
      return getattr(inputs, '_keras_mask', None)

  @property
  def output_mask(self):
    output = self.output
    if isinstance(output, list):
      return [getattr(x, '_keras_mask', None) for x in output]
    else:
      return getattr(output, '_keras_mask', None)

  def get_input_shape_at(self, node_index):
    return self._get_node_attribute_at_index(node_index, 'input_shapes',
                                             'input shape')

  def get_output_shape_at(self, node_index):
    return self._get_node_attribute_at_index(node_index, 'output_shapes',
                                             'output shape')

  def get_input_at(self, node_index):
    return self._get_node_attribute_at_index(node_index, 'input_tensors',
                                             'input')

  def get_output_at(self, node_index):
    return self._get_node_attribute_at_index(node_index, 'output_tensors',
                                             'output')

  @property
  def input(self):
    if not self._inbound_nodes:
      raise AttributeError('Layer ' + self.name +
                           ' is not connected, no input to return.')
    return self._get_node_attribute_at_index(0, 'input_tensors', 'input')

  @property
  def output(self):
    if not self._inbound_nodes:
      raise AttributeError('Layer ' + self.name + ' has no inbound nodes.')
    return self._get_node_attribute_at_index(0, 'output_tensors', 'output')

  @property
  def input_shape(self):
    if not self._inbound_nodes:
      raise AttributeError('The layer has never been called '
                           'and thus has no defined input shape.')
    all_input_shapes = set(
        [str(node.input_shapes) for node in self._inbound_nodes])
    if len(all_input_shapes) == 1:
      input_shapes = self._inbound_nodes[0].input_shapes
      if len(input_shapes) == 1:
        return tuple(tensor_shape.TensorShape(input_shapes[0]).as_list())
      else:
        return [
            tuple(tensor_shape.TensorShape(shape).as_list())
            for shape in input_shapes
        ]
    else:
      raise AttributeError('The layer "' + str(self.name) +
                           ' has multiple inbound nodes, '
                           'with different input shapes. Hence '
                           'the notion of "input shape" is '
                           'ill-defined for the layer. '
                           'Use `get_input_shape_at(node_index)` '
                           'instead.')

  def count_params(self):
    if not self.built:
      if self.__class__.__name__ == 'Sequential':
        self.build()  # pylint: disable=no-value-for-parameter
      else:
        raise ValueError('You tried to call `count_params` on ' + self.name +
                         ', but the layer isn\'t built. '
                         'You can build it manually via: `' + self.name +
                         '.build(batch_input_shape)`.')
    weight_shapes = [w.shape.as_list() for w in self.weights]
    return int(sum([np.prod(w) for w in weight_shapes]))

  @property
  def output_shape(self):
    if not self._inbound_nodes:
      raise AttributeError('The layer has never been called '
                           'and thus has no defined output shape.')
    all_output_shapes = set(
        [str(node.output_shapes) for node in self._inbound_nodes])
    if len(all_output_shapes) == 1:
      output_shapes = self._inbound_nodes[0].output_shapes
      if len(output_shapes) == 1:
        return tuple(tensor_shape.TensorShape(output_shapes[0]).as_list())
      else:
        return [
            tuple(tensor_shape.TensorShape(shape).as_list())
            for shape in output_shapes
        ]
    else:
      raise AttributeError('The layer "%s"'
                           ' has multiple inbound nodes, '
                           'with different output shapes. Hence '
                           'the notion of "output shape" is '
                           'ill-defined for the layer. '
                           'Use `get_output_shape_at(node_index)` '
                           'instead.' % self.name)

  @property
  @doc_controls.do_not_doc_inheritable
  def inbound_nodes(self):
    return self._inbound_nodes

  @property
  @doc_controls.do_not_doc_inheritable
  def outbound_nodes(self):
    return self._outbound_nodes

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

      # Check ndim.
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
      # Check dtype.
      if spec.dtype is not None:
        if x.dtype != spec.dtype:
          raise ValueError('Input ' + str(input_index) + ' of layer ' +
                           self.name + ' is incompatible with the layer: '
                           'expected dtype=' + str(spec.dtype) +
                           ', found dtype=' + str(x.dtype))
      # Check specific shape axes.
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
      # Check shape.
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

  def set_weights(self, weights):
    params = self.weights
    if len(params) != len(weights):
      raise ValueError('You called `set_weights(weights)` on layer "' +
                       self.name + '" with a  weight list of length ' +
                       str(len(weights)) + ', but the layer was expecting ' +
                       str(len(params)) + ' weights. Provided weights: ' +
                       str(weights)[:50] + '...')
    if not params:
      return
    weight_value_tuples = []
    param_values = backend.batch_get_value(params)
    for pv, p, w in zip(param_values, params, weights):
      if pv.shape != w.shape:
        raise ValueError('Layer weight shape ' + str(pv.shape) +
                         ' not compatible with '
                         'provided weight shape ' + str(w.shape))
      weight_value_tuples.append((p, w))
    backend.batch_set_value(weight_value_tuples)

  def get_weights(self):
    params = self.weights
    return backend.batch_get_value(params)

  def get_config(self):
    config = {'name': self.name, 'trainable': self.trainable}
    if hasattr(self, '_batch_input_shape'):
      config['batch_input_shape'] = self._batch_input_shape
    if hasattr(self, 'dtype'):
      config['dtype'] = self.dtype
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)


def to_snake_case(name):
  intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
  insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
  # If the class is private the name starts with "_" which is not secure
  # for creating scopes. We prefix the name with "private" in this case.
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
  # TODO(fchollet): refactor this into backend.
  graph = ops.get_default_graph()
  name_uid_map = backend.PER_GRAPH_LAYER_NAME_UIDS.get(graph, None)
  if name_uid_map is None:
    name_uid_map = collections_lib.defaultdict(int)
    backend.PER_GRAPH_LAYER_NAME_UIDS[graph] = name_uid_map
  return name_uid_map

class CallConvention(enum.Enum):
  # The Layer takes inputs as its first argument, named "inputs" for
  # compatibility with the signature of Layer.__call__. This is the mode assumed
  # for Layers which are not subclassed Models.
  EXPLICIT_INPUTS_ARGUMENT = 1
  # The Layer takes a single positional argument, not named "inputs". It's
  # treated like an "inputs" argument.
  SINGLE_POSITIONAL_ARGUMENT = 2
  # The Layer has multiple positional arguments to which its inputs should be
  # bound.
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


def collect_previous_mask(input_tensors):
  input_tensors = nest.flatten(input_tensors)
  masks = []
  for x in input_tensors:
    if hasattr(x, '_keras_mask'):
      mask = x._keras_mask  # pylint: disable=protected-access
      masks.append(mask)
    else:
      masks.append(None)
  if len(masks) == 1:
    return masks[0]
  return masks

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

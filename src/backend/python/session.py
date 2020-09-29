from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import re
import threading
import warnings

import numpy as np


from backend.python.framework import errors_impl as errors
from backend.util import compat
from backend.util import nest
from backend.python import pywrap_backend as tf_session
from backend.python.ops import session_ops
from backend.python.framework import ops
from backend.python.framework import sparse_tensor



def _get_feeds_for_indexed_slices(feed, feed_val):
  return list(
      zip([feed.values, feed.indices] if feed.dense_shape is None else
          [feed.values, feed.indices, feed.dense_shape], feed_val))

_REGISTERED_EXPANSIONS = [
    (sparse_tensor.SparseTensor,
     lambda fetch: (
         [fetch.indices, fetch.values, fetch.dense_shape],
         lambda fetched_vals: sparse_tensor.SparseTensorValue(*fetched_vals)),
     lambda feed, feed_val: list(zip(
         [feed.indices, feed.values, feed.dense_shape], feed_val)),
     lambda feed: [feed.indices, feed.values, feed.dense_shape]),
    (ops.IndexedSlices,
     lambda fetch: (
         [fetch.values, fetch.indices] if fetch.dense_shape is None
         else [fetch.values, fetch.indices, fetch.dense_shape],
         _get_indexed_slices_value_from_fetches),
     _get_feeds_for_indexed_slices,
     lambda feed: [feed.values, feed.indices] if feed.dense_shape is None
     else [feed.values, feed.indices, feed.dense_shape]),
    (object,
     lambda fetch: ([fetch], lambda fetched_vals: fetched_vals[0]),
     lambda feed, feed_val: [(feed, feed_val)],
     lambda feed: [feed])]

def _convert_to_numpy_obj(numpy_dtype, obj):
  return numpy_dtype(obj) if numpy_dtype is not object else str(obj)

def _is_attrs_instance(obj):
  return getattr(obj.__class__, '__attrs_attrs__', None) is not None

class _FetchMapper(object):
  @staticmethod
  def for_fetch(fetch):
    if isinstance(fetch, (list, tuple)):
      return _ListFetchMapper(fetch)
    elif isinstance(fetch, collections.Mapping):
      return _DictFetchMapper(fetch)
    elif _is_attrs_instance(fetch):
      return _AttrsFetchMapper(fetch)
    else:
      for tensor_type, fetch_fn, _, _ in _REGISTERED_EXPANSIONS:
        if isinstance(fetch, tensor_type):
          fetches, contraction_fn = fetch_fn(fetch)
          return _ElementFetchMapper(fetches, contraction_fn)

class _ElementFetchMapper(_FetchMapper):
  def __init__(self, fetches, contraction_fn):
    self._unique_fetches = []
    for fetch in fetches:
      self._unique_fetches.append(ops.get_default_graph().as_graph_element(
            fetch, allow_tensor=True, allow_operation=True))
    self._contraction_fn = contraction_fn

  def unique_fetches(self):
    return self._unique_fetches

  def build_results(self, values):
    return self._contraction_fn(values)

def _uniquify_fetches(fetch_mappers):
  unique_fetches = []
  value_indices = []
  seen_fetches = {}
  for m in fetch_mappers:
    m_value_indices = []
    for f in m.unique_fetches():
      j = seen_fetches.get(f)
      if j is None:
        j = len(seen_fetches)
        seen_fetches[f] = j
        unique_fetches.append(f)
      m_value_indices.append(j)
    value_indices.append(m_value_indices)
  return unique_fetches, value_indices

class _ListFetchMapper(_FetchMapper):
  def __init__(self, fetches):
    self._fetch_type = type(fetches)
    self._mappers = [_FetchMapper.for_fetch(fetch) for fetch in fetches]
    self._unique_fetches, self._value_indices = _uniquify_fetches(self._mappers)

  def unique_fetches(self):
    return self._unique_fetches

  def build_results(self, values):
    results = []
    for m, vi in zip(self._mappers, self._value_indices):
      results.append(m.build_results([values[j] for j in vi]))
    if issubclass(self._fetch_type, list):
      return results
    elif self._fetch_type == tuple:
      return tuple(results)
    else:
      return self._fetch_type(*results)

class _FetchHandler(object):
  def __init__(self, graph, fetches, feeds, feed_handles=None):
    with graph.as_default():
      self._fetch_mapper = _FetchMapper.for_fetch(fetches)
    self._fetches = []
    self._targets = []
    self._feeds = feeds
    self._feed_handles = feed_handles or {}
    self._ops = []
    self._fetch_handles = {}
    for fetch in self._fetch_mapper.unique_fetches():
      if isinstance(fetch, ops.Operation):
        self._assert_fetchable(graph, fetch)
        self._targets.append(fetch)
        self._ops.append(True)
      else:
        self._assert_fetchable(graph, fetch.op)
        self._fetches.append(fetch)
        self._ops.append(False)
      if (isinstance(fetch, ops.Tensor) and
          (fetch.op.type == 'GetSessionHandle' or
           fetch.op.type == 'GetSessionHandleV2')):
        self._fetch_handles[fetch] = fetch.op.inputs[0].dtype
    self._final_fetches = [x for x in self._fetches if x not in feeds]

  def _assert_fetchable(self, graph, op):
    if not graph.is_fetchable(op):
      raise ValueError(
          'Operation %r has been marked as not fetchable.' % op.name)

  def fetches(self):
    return self._final_fetches

  def targets(self):
    return self._targets

  def build_results(self, session, tensor_values):
    full_values = []

    i = 0
    j = 0
    for is_op in self._ops:
      if is_op:
        full_values.append(None)
      else:
        if self._fetches[i] in self._feed_handles:
          value = self._feed_handles[self._fetches[i]].eval()
        else:
          value = self._feeds.get(self._fetches[i])
        if value is None:
          value = tensor_values[j]
          j += 1
        dtype = self._fetch_handles.get(self._fetches[i])
        if dtype:
          full_values.append(session_ops.TensorHandle(value, dtype, session))
        else:
          full_values.append(value)
        i += 1
    assert j == len(tensor_values)
    return self._fetch_mapper.build_results(full_values)

class _DeviceAttributes(object):

  def __init__(self, name, device_type, memory_limit_bytes, incarnation):
    self._name = device.canonical_name(name)
    self._device_type = device_type
    self._memory_limit_bytes = memory_limit_bytes
    self._incarnation = incarnation

  @property
  def name(self):
    return self._name

  @property
  def device_type(self):
    return self._device_type

  @property
  def memory_limit_bytes(self):
    return self._memory_limit_bytes

  @property
  def incarnation(self):
    return self._incarnation

  def __repr__(self):
    return '_DeviceAttributes(%s, %s, %d, %d)' % (
        self.name,
        self.device_type,
        self.memory_limit_bytes,
        self.incarnation,
    )

class BaseSession(object):
  def __init__(self, target='', graph=None, config=None):
    if graph is None:
      self._graph = ops.get_default_graph()
    else:
      if not isinstance(graph, ops.Graph):
        raise TypeError('graph must be a tf.Graph, but got %s' % type(graph))
      self._graph = graph

    self._opened = False
    self._closed = False

    self._current_version = 0
    self._extend_lock = threading.Lock()
    if target is not None:
      try:
        self._target = compat.as_bytes(target)
      except TypeError:
        raise TypeError('target must be a string, but got %s' % type(target))
    else:
      self._target = None

    self._delete_lock = threading.Lock()
    self._dead_handles = []

    if config is not None:
      if not isinstance(config, config_pb2.ConfigProto):
        raise TypeError(
            'config must be a tf.ConfigProto, but got %s' % type(config))
      self._config = config
      self._add_shapes = config.graph_options.infer_shapes
    else:
      self._config = None
      self._add_shapes = False

    self._session = None
    opts = tf_session.TF_NewSessionOptions(target=self._target, config=config)
    try:
     
      self._session = tf_session.TF_NewSessionRef(self._graph._c_graph, opts)
      
    finally:
      tf_session.TF_DeleteSessionOptions(opts)

  def __del__(self):
    try:
      self.close()
    except Exception: 
      pass
    if self._session is not None:
      try:
        tf_session.TF_DeleteSession(self._session)
      except AttributeError:
        pass
      self._session = None

  @property
  def graph(self):
    return self._graph

  def as_default(self):
    return ops.default_session(self)

  def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
    options_ptr = tf_session.TF_NewBufferFromString(
        compat.as_bytes(options.SerializeToString())) if options else None
    run_metadata_ptr = tf_session.TF_NewBuffer() if run_metadata else None

    try:
      result = self._run(None, fetches, feed_dict, options_ptr,
                         run_metadata_ptr)
      if run_metadata:
        proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)
        run_metadata.ParseFromString(compat.as_bytes(proto_data))
    finally:
      if run_metadata_ptr:
        tf_session.TF_DeleteBuffer(run_metadata_ptr)
      if options:
        tf_session.TF_DeleteBuffer(options_ptr)
    return result

  def _run(self, handle, fetches, feed_dict, options, run_metadata):

    def _feed_fn(feed, feed_val):
      for tensor_type, _, feed_fn, _ in _REGISTERED_EXPANSIONS:
        if isinstance(feed, tensor_type):
          return feed_fn(feed, feed_val)
      raise TypeError('Feed argument %r has invalid type %r' % (feed,
                                                                type(feed)))

    if self._closed:
      raise RuntimeError('Attempted to use a closed Session.')
    if self.graph.version == 0:
      raise RuntimeError('The Session graph is empty.  Add operations to the '
                         'graph before calling run().')


    feed_dict_tensor = {}
    feed_map = {}


    feed_handles = {}
    if feed_dict:
      feed_dict = nest.flatten_dict_items(feed_dict)
      for feed, feed_val in feed_dict.items():
        for subfeed, subfeed_val in _feed_fn(feed, feed_val):
          try:
            subfeed_t = self.graph.as_graph_element(
                subfeed, allow_tensor=True, allow_operation=False)
          except Exception as e:
            raise TypeError(
                'Cannot interpret feed_dict key as Tensor: ' + e.args[0])

          if isinstance(subfeed_val, ops.Tensor):
            raise TypeError('The value of a feed cannot be a tf.Tensor object. '
                            'Acceptable feed values include Python scalars, '
                            'strings, lists, numpy ndarrays, or TensorHandles.'
                            'For reference, the tensor object was ' +
                            str(feed_val) + ' which was passed to the '
                            'feed with key ' + str(feed) + '.')

          subfeed_dtype = subfeed_t.dtype.as_numpy_dtype
          if isinstance(subfeed_val, int) and _convert_to_numpy_obj(
              subfeed_dtype, subfeed_val) != subfeed_val:
            raise TypeError(
                'Type of feed value ' + str(subfeed_val) + ' with type ' + str(
                    type(subfeed_val)) +
                ' is not compatible with Tensor type ' + str(subfeed_dtype) +
                '. Try explicitly setting the type of the feed tensor'
                ' to a larger type (e.g. int64).')

          is_tensor_handle_feed = isinstance(subfeed_val,
                                             session_ops.TensorHandle)
          if is_tensor_handle_feed:
            np_val = subfeed_val.to_numpy_array()
            feed_handles[subfeed_t] = subfeed_val
          else:
            np_val = np.asarray(subfeed_val, dtype=subfeed_dtype)

          if (not is_tensor_handle_feed and
              not subfeed_t.get_shape().is_compatible_with(np_val.shape)):
            raise ValueError('Cannot feed value of shape %r for Tensor %r, '
                             'which has shape %r' %
                             (np_val.shape, subfeed_t.name,
                              str(subfeed_t.get_shape())))
          if not self.graph.is_feedable(subfeed_t):
            raise ValueError('Tensor %s may not be fed.' % subfeed_t)

          feed_dict_tensor[subfeed_t] = np_val
          feed_map[compat.as_bytes(subfeed_t.name)] = (subfeed_t, subfeed_val)

    fetch_handler = _FetchHandler(
        self._graph, fetches, feed_dict_tensor, feed_handles=feed_handles)


    _ = self._update_with_movers(feed_dict_tensor, feed_map)
    final_fetches = fetch_handler.fetches()
    final_targets = fetch_handler.targets()
    if final_fetches or final_targets or (handle and feed_dict_tensor):
      results = self._do_run(handle, final_targets, final_fetches,
                             feed_dict_tensor, options, run_metadata)
    else:
      results = []
    return fetch_handler.build_results(self, results)

  def make_callable(self, fetches, feed_list=None, accept_options=False):
    if feed_list is not None:
      if not isinstance(feed_list, (list, tuple)):
        raise TypeError('`feed_list` must be a list or tuple.')
      def _generic_run(*feed_args, **kwargs):
        feed_dict = {
            feed: feed_val
            for feed, feed_val in zip(feed_list, feed_args)
        }
        return self.run(fetches, feed_dict=feed_dict, **kwargs)

      return _generic_run

    self._extend_graph()

    fetch_handler = _FetchHandler(self._graph, fetches, {})
   
    fetch_list = [t._as_tf_output() for t in fetch_handler.fetches()]
    target_list = [op._c_op for op in fetch_handler.targets()]
    

    def _callable_template_with_options_and_metadata(fetch_list,
                                                     target_list,
                                                     fetch_handler,
                                                     options=None,
                                                     run_metadata=None):
      options_ptr = tf_session.TF_NewBufferFromString(
          compat.as_bytes(options.SerializeToString())) if options else None
      run_metadata_ptr = tf_session.TF_NewBuffer() if run_metadata else None
      try:
        results = self._call_tf_sessionrun(
            options_ptr, {}, fetch_list, target_list, run_metadata_ptr)
        if fetch_handler:
          results = fetch_handler.build_results(self, results)
        else:
          results = results[0] if results else None
        if run_metadata:
          proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)
          run_metadata.ParseFromString(compat.as_bytes(proto_data))
      finally:
        if run_metadata_ptr:
          tf_session.TF_DeleteBuffer(run_metadata_ptr)
        if options:
          tf_session.TF_DeleteBuffer(options_ptr)
      return results

    if accept_options:
      return functools.partial(_callable_template_with_options_and_metadata,
                               fetch_list, target_list, fetch_handler)
    elif isinstance(fetches, ops.Operation):
      assert not fetch_list
      assert len(target_list) == 1

      def _single_operation_run():
        self._call_tf_sessionrun(None, {}, [], target_list, None)

      return _single_operation_run
    elif isinstance(fetches, ops.Tensor):
      assert len(fetch_list) == 1
      assert not target_list

      def _single_tensor_run():
        results = self._call_tf_sessionrun(None, {}, fetch_list, [], None)
        return results[0]

      return _single_tensor_run
    else:
      def _fetch_handler_run():
        results = self._call_tf_sessionrun(
            None, {}, fetch_list, target_list, None)
        return fetch_handler.build_results(self, results)

      return _fetch_handler_run

  _NODEDEF_NAME_RE = re.compile(
      r'\[\[(Node: )?(\{\{node )?([^\} ]*)(\}\})?\s*=')

  def _do_run(self, handle, target_list, fetch_list, feed_dict, options,
              run_metadata):
   
    feeds = dict((t._as_tf_output(), v) for t, v in feed_dict.items())
    fetches = [t._as_tf_output() for t in fetch_list]
    targets = [op._c_op for op in target_list]
    

    def _run_fn(feed_dict, fetch_list, target_list, options, run_metadata):
      self._extend_graph()
      return self._call_tf_sessionrun(
          options, feed_dict, fetch_list, target_list, run_metadata)

    def _prun_fn(handle, feed_dict, fetch_list):
      if target_list:
        raise RuntimeError('partial_run() requires empty target_list.')
      return self._call_tf_sessionprun(handle, feed_dict, fetch_list)

    if handle is None:
      return self._do_call(_run_fn, feeds, fetches, targets, options,
                           run_metadata)
    else:
      return self._do_call(_prun_fn, handle, feeds, fetches)

  def _do_call(self, fn, *args):
    try:
      return fn(*args)
    except errors.OpError as e:
      message = compat.as_text(e.message)
      m = BaseSession._NODEDEF_NAME_RE.search(message)
      node_def = None
      op = None
      if m is not None:
        node_name = m.group(3)
        try:
          op = self._graph.get_operation_by_name(node_name)
          node_def = op.node_def
        except KeyError:
          pass

  def _extend_graph(self):
    with self._graph._session_run_lock(): 
      tf_session.ExtendSession(self._session)

  _DEAD_HANDLES_THRESHOLD = 10

  def _update_with_movers(self, feed_dict, feed_map):
    handle_movers = []
    for feed_name, val in feed_map.items():
      mover = session_ops._get_handle_mover(self.graph, *val)
      if mover:
        handle_movers.append((feed_name, val[1], mover))
    if not handle_movers:
      return []
    else:
      feeds = {}
      fetches = []
      for _, handle, mover in handle_movers:
        feeds[mover[0]] = handle
        fetches.append(mover[1])
      handles = self.run(fetches, feed_dict=feeds)
      for handle_mover, handle in zip(handle_movers, handles):
        np_val = np.array(handle.handle, dtype=np.object)
        feed_name = handle_mover[0]
        feed_tensor = feed_map[feed_name][0]
        feed_dict[feed_tensor] = np_val
      return handles

  def _call_tf_sessionrun(self, options, feed_dict, fetch_list, target_list,
                          run_metadata):
    return tf_session.TF_SessionRun_wrapper(
        self._session, options, feed_dict, fetch_list, target_list,
        run_metadata)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import traceback

def CheckInputFromValidContext(op, input_op):
  op_ctxt = op._get_control_flow_context()  
  input_ctxt = GetOutputContext(input_op)
  valid = False

  if not input_ctxt:
    valid = True
  elif op_ctxt is input_ctxt:
    valid = True
  else:
    while_ctxt = GetContainingWhileContext(op_ctxt)
    input_while_ctxt = GetContainingWhileContext(input_ctxt)

    if while_ctxt is None:
      if input_while_ctxt is None:
        valid = True
      if IsLoopEnter(op):
        valid = True
      if IsSwitch(op):
        valid = True
    elif IsContainingContext(while_ctxt, input_while_ctxt):
      valid = True
    elif (while_ctxt.grad_state and
          IsContainingContext(while_ctxt.grad_state.forward_context,
                              input_while_ctxt)):
      valid = True
    elif (while_ctxt.grad_state and
          while_ctxt.grad_state.forward_context is
          input_while_ctxt._outer_context):  
      valid = True
    elif (input_while_ctxt.grad_state and
          input_while_ctxt.grad_state.forward_context is while_ctxt):
      valid = True
    elif (input_while_ctxt.grad_state and
          input_ctxt.grad_state.forward_context.grad_state and
          input_ctxt.grad_state.forward_context.grad_state.forward_context is
          while_ctxt):
      valid = True

  if not valid:
    if while_ctxt:
      error_msg = (
          "Cannot use '%s' as input to '%s' because they are in different while"
          " loops." % (op.name, input_op.name))
    else:
      error_msg = (
          "Cannot use '%s' as input to '%s' because '%s' is in a while loop."
          % (input_op.name, op.name, input_op.name))

    log_msg = error_msg
    log_msg += "\n\n%s while context: %s" % (op.name, while_ctxt)
    log_msg += "\n%s while context: %s" % (input_op.name, input_while_ctxt)
    log_msg += "\n\nTraceback for %s:\n%s\nTraceback for %s:\n%s\n" % (
        op.name, "".join(traceback.format_list(op.traceback)),
        input_op.name, "".join(traceback.format_list(input_op.traceback)))
    logging.info(log_msg)
    raise ValueError(error_msg + " See info log for more details.")

def GetOutputContext(op):
  ctxt = op._get_control_flow_context()  
  if ctxt is not None and IsLoopExit(op):
    ctxt = ctxt.outer_context
  return ctxt

def IsLoopExit(op):
  return op.type == "Exit" or op.type == "RefExit"

def GetContainingWhileContext(ctxt, stop_ctxt=None):
  while ctxt:
    if ctxt.IsWhileContext() or ctxt == stop_ctxt: return ctxt
    ctxt = ctxt.outer_context
  return None

def IsLoopEnter(op):
  return op.type == "Enter" or op.type == "RefEnter"

def IsSwitch(op):
  return op.type == "Switch" or op.type == "RefSwitch"

def IsLoopConstantEnter(op):
  return IsLoopEnter(op) and op.get_attr("is_constant")

def IsLoopSwitch(op):
  if IsSwitch(op):
    ctxt = op._get_control_flow_context()  
    return ctxt is not None and ctxt.IsWhileContext() and not IsCondSwitch(op)
  return False
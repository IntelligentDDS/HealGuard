/**
 * @name Repository CFG and DFG
 * @description Exports explicit repository control-flow, data-flow, call, return, exception, function-summary, and sensitive-sink edges.
 * @kind table
 * @id agenticheal/python-healing-capabilities
 */
import python
import meta.alerts.Sinks
import semmle.python.frameworks.data.ModelsAsData
import semmle.python.dataflow.new.DataFlow
import semmle.python.types.FunctionObject

predicate projectLocation(Location location) {
  exists(location.getFile().getRelativePath())
}

predicate sensitiveSink(DataFlow::Node sink, string kind) {
  projectLocation(sink.getLocation()) and
  (sink = taintSink(kind) or ModelOutput::sinkNode(sink, kind))
}

predicate reachesSensitiveSink(FunctionObject callable, DataFlow::Node sink, string kind) {
  projectLocation(callable.getFunction().getLocation()) and
  sensitiveSink(sink, kind) and
  exists(FunctionObject sinkCallable |
    sinkCallable.getFunction() = sink.asCfgNode().getScope() and
    projectLocation(sinkCallable.getFunction().getLocation()) and
    (callable = sinkCallable or callable.getACallee+() = sinkCallable)
  )
}

string scopeName(Scope scope) {
  result = scope.toString()
}

string callableName(FunctionObject callable) {
  result = callable.getFunction().getScope().toString() + "." + callable.getFunction().getName()
}

predicate locationInfo(Location location, string file, int startLine, int startColumn,
  int endLine, int endColumn) {
  location.hasLocationInfo(file, startLine, startColumn, endLine, endColumn)
}

predicate enclosingStatement(AstNode node, Stmt statement) {
  node = statement
  or
  node.getParentNode+() = statement and
  not exists(Stmt inner |
    node.getParentNode+() = inner and inner.getParentNode+() = statement
  )
}

string dataName(DataFlow::Node node) {
  exists(Name name | node.asCfgNode().getNode() = name and result = name.getId())
  or
  not exists(Name name | node.asCfgNode().getNode() = name) and result = ""
}

from string recordType, string edgeType,
  string sourceFunction, string sourceVariable, string sourceFile,
  int sourceLine, int sourceColumn, int sourceEndLine, int sourceEndColumn,
  string targetFunction, string targetVariable, string targetFile,
  int targetLine, int targetColumn, int targetEndLine, int targetEndColumn,
  string detail
where
  exists(ControlFlowNode source, ControlFlowNode target, Stmt sourceStatement, Stmt targetStatement,
    Location sourceLocation, Location targetLocation |
    recordType = "cfg_edge" and edgeType = "cfg_next" and target = source.getASuccessor() and
    enclosingStatement(source.getNode(), sourceStatement) and enclosingStatement(target.getNode(), targetStatement) and
    sourceFunction = scopeName(source.getScope()) and targetFunction = scopeName(target.getScope()) and
    sourceVariable = "" and targetVariable = "" and
    sourceLocation = sourceStatement.getLocation() and targetLocation = targetStatement.getLocation() and
    projectLocation(sourceLocation) and projectLocation(targetLocation) and
    locationInfo(sourceLocation, sourceFile, sourceLine, sourceColumn, sourceEndLine, sourceEndColumn) and
    locationInfo(targetLocation, targetFile, targetLine, targetColumn, targetEndLine, targetEndColumn) and
    detail = source.toString() + " -> " + target.toString()
  )
  or
  exists(ControlFlowNode source, ControlFlowNode target, Stmt sourceStatement, Stmt targetStatement,
    Location sourceLocation, Location targetLocation |
    recordType = "cfg_edge" and edgeType = "cfg_exception" and target = source.getAnExceptionalSuccessor() and
    enclosingStatement(source.getNode(), sourceStatement) and enclosingStatement(target.getNode(), targetStatement) and
    sourceFunction = scopeName(source.getScope()) and targetFunction = scopeName(target.getScope()) and
    sourceVariable = "" and targetVariable = "" and
    sourceLocation = sourceStatement.getLocation() and targetLocation = targetStatement.getLocation() and
    projectLocation(sourceLocation) and projectLocation(targetLocation) and
    locationInfo(sourceLocation, sourceFile, sourceLine, sourceColumn, sourceEndLine, sourceEndColumn) and
    locationInfo(targetLocation, targetFile, targetLine, targetColumn, targetEndLine, targetEndColumn) and
    detail = source.toString() + " -> " + target.toString()
  )
  or
  exists(DataFlow::Node source, DataFlow::Node target, Stmt sourceStatement, Stmt targetStatement,
    Location sourceLocation, Location targetLocation |
    recordType = "dfg_edge" and edgeType = "dfg_local" and DataFlow::localFlowStep(source, target) and
    enclosingStatement(source.asCfgNode().getNode(), sourceStatement) and
    enclosingStatement(target.asCfgNode().getNode(), targetStatement) and
    sourceFunction = scopeName(source.getScope()) and targetFunction = scopeName(target.getScope()) and
    sourceVariable = dataName(source) and targetVariable = dataName(target) and
    sourceLocation = sourceStatement.getLocation() and targetLocation = targetStatement.getLocation() and
    projectLocation(sourceLocation) and projectLocation(targetLocation) and
    locationInfo(sourceLocation, sourceFile, sourceLine, sourceColumn, sourceEndLine, sourceEndColumn) and
    locationInfo(targetLocation, targetFile, targetLine, targetColumn, targetEndLine, targetEndColumn) and
    detail = source.toString() + " -> " + target.toString()
  )
  or
  exists(FunctionObject caller, FunctionObject callee, CallNode call, Location sourceLocation,
    Location targetLocation |
    recordType = "cfg_edge" and edgeType = "cfg_call" and caller.getFunction() = call.getScope() and
    callee.getACall() = call and sourceFunction = callableName(caller) and
    targetFunction = callableName(callee) and sourceVariable = call.toString() and
    targetVariable = callee.getFunction().toString() and sourceLocation = call.getLocation() and
    targetLocation = callee.getFunction().getLocation() and
    projectLocation(sourceLocation) and projectLocation(targetLocation) and
    locationInfo(sourceLocation, sourceFile, sourceLine, sourceColumn, sourceEndLine, sourceEndColumn) and
    locationInfo(targetLocation, targetFile, targetLine, targetColumn, targetEndLine, targetEndColumn) and
    detail = callableName(callee)
  )
  or
  exists(FunctionObject callee, CallNode call, int index, DataFlow::Node source, DataFlow::Node target,
    Location sourceLocation, Location targetLocation |
    recordType = "dfg_edge" and edgeType = "dfg_argument_parameter" and callee.getACall() = call and
    source.asCfgNode() = call.getArg(index) and
    target = DataFlow::parameterNode(callee.getFunction().getArg(index)) and
    sourceFunction = scopeName(call.getScope()) and targetFunction = callableName(callee) and
    sourceVariable = source.toString() and targetVariable = callee.getFunction().getArg(index).getName() and
    sourceLocation = source.getLocation() and targetLocation = target.getLocation() and
    projectLocation(sourceLocation) and projectLocation(targetLocation) and
    locationInfo(sourceLocation, sourceFile, sourceLine, sourceColumn, sourceEndLine, sourceEndColumn) and
    locationInfo(targetLocation, targetFile, targetLine, targetColumn, targetEndLine, targetEndColumn) and
    detail = "argument " + index.toString()
  )
  or
  exists(FunctionObject callee, CallNode call, ControlFlowNode source, Location sourceLocation,
    Location targetLocation |
    recordType = "dfg_edge" and edgeType = "dfg_return_call" and callee.getACall() = call and
    source = callee.getFunction().getReturnNode() and sourceFunction = callableName(callee) and
    targetFunction = scopeName(call.getScope()) and sourceVariable = "return" and
    targetVariable = call.toString() and sourceLocation = source.getLocation() and
    targetLocation = call.getLocation() and
    projectLocation(sourceLocation) and projectLocation(targetLocation) and
    locationInfo(sourceLocation, sourceFile, sourceLine, sourceColumn, sourceEndLine, sourceEndColumn) and
    locationInfo(targetLocation, targetFile, targetLine, targetColumn, targetEndLine, targetEndColumn) and
    detail = callableName(callee)
  )
  or
  exists(DataFlow::AttrWrite write, Location location |
    recordType = "dfg_edge" and edgeType = "dfg_attribute_write" and
    sourceFunction = scopeName(write.getScope()) and targetFunction = sourceFunction and
    sourceVariable = write.getObject().toString() and targetVariable = write.getAttributeName() and
    location = write.getLocation() and projectLocation(location) and
    locationInfo(location, sourceFile, sourceLine, sourceColumn, sourceEndLine, sourceEndColumn) and
    targetFile = sourceFile and targetLine = sourceLine and targetColumn = sourceColumn and
    targetEndLine = sourceEndLine and targetEndColumn = sourceEndColumn and
    detail = "attribute_write"
  )
  or
  exists(FunctionObject source, DataFlow::Node sink, string kind, Location sourceLocation,
    Location targetLocation |
    recordType = "function_summary" and edgeType = "summary_sensitive_sink" and
    reachesSensitiveSink(source, sink, kind) and sourceFunction = callableName(source) and
    targetFunction = scopeName(sink.getScope()) and sourceVariable = "return" and
    targetVariable = sink.toString() and sourceLocation = source.getFunction().getLocation() and
    targetLocation = sink.getLocation() and
    projectLocation(sourceLocation) and projectLocation(targetLocation) and
    locationInfo(sourceLocation, sourceFile, sourceLine, sourceColumn, sourceEndLine, sourceEndColumn) and
    locationInfo(targetLocation, targetFile, targetLine, targetColumn, targetEndLine, targetEndColumn) and
    detail = kind
  )
  or
  exists(DataFlow::Node sink, string kind, Location location |
    recordType = "sink" and edgeType = "confirmed_sink_candidate" and sensitiveSink(sink, kind) and
    sourceFunction = scopeName(sink.getScope()) and targetFunction = sourceFunction and
    sourceVariable = sink.toString() and targetVariable = sourceVariable and location = sink.getLocation() and
    projectLocation(location) and locationInfo(location, sourceFile, sourceLine, sourceColumn, sourceEndLine, sourceEndColumn) and
    targetFile = sourceFile and targetLine = sourceLine and targetColumn = sourceColumn and
    targetEndLine = sourceEndLine and targetEndColumn = sourceEndColumn and detail = kind
  )
select recordType, edgeType, sourceFunction, sourceVariable, sourceFile,
  sourceLine, sourceColumn, sourceEndLine, sourceEndColumn,
  targetFunction, targetVariable, targetFile,
  targetLine, targetColumn, targetEndLine, targetEndColumn, detail

/**
 * @name Compact sensitive graph
 * @description Exports selected sinks and direct semantic edges without expanding nodes by reachable sink.
 * @kind table
 * @id agenticheal/python-sensitive-models
 */
import python
import meta.alerts.Sinks
import semmle.python.frameworks.data.ModelsAsData
import semmle.python.dataflow.new.DataFlow
import semmle.python.GuardedControlFlow
import semmle.python.types.FunctionObject

external predicate candidateSink(
  string file, int startLine, int startColumn, int endLine, int endColumn
);

predicate locationInfo(Location location, string file, int startLine, int startColumn, int endLine, int endColumn) {
  location.hasLocationInfo(file, startLine, startColumn, endLine, endColumn)
}

predicate selectedSink(DataFlow::Node sink, string kind) {
  exists(Location location, string file, int startLine, int startColumn, int endLine, int endColumn,
    int candidateStartLine, int candidateStartColumn, int candidateEndLine, int candidateEndColumn |
    (sink = taintSink(kind) or ModelOutput::sinkNode(sink, kind)) and
    location = sink.getLocation() and
    locationInfo(location, file, startLine, startColumn, endLine, endColumn) and
    candidateSink(file, candidateStartLine, candidateStartColumn, candidateEndLine, candidateEndColumn) and
    candidateStartLine <= startLine and endLine <= candidateEndLine and
    (candidateStartLine < startLine or candidateStartColumn <= startColumn) and
    (endLine < candidateEndLine or endColumn <= candidateEndColumn)
  )
}

predicate sinkRelevantScope(Scope scope) {
  exists(DataFlow::Node sink, string kind | selectedSink(sink, kind) and scope = sink.getScope())
  or
  exists(FunctionObject callee |
    callee.getACall().getScope() = scope and sinkRelevantScope(callee.getFunction())
  )
}

predicate directControlEdge(ControlFlowNode source, ControlFlowNode target, string detail) {
  exists(ConditionBlock condition, boolean branch |
    source = condition.getLastNode() and
    condition.controls(target.getBasicBlock(), branch) and
    (branch = true and detail = "true" or branch = false and detail = "false")
  )
}

predicate directCallEdge(ControlFlowNode source, ControlFlowNode target, string detail) {
  exists(FunctionObject callee |
    source = callee.getACall() and
    target = callee.getFunction().getEntryNode().getASuccessor() and
    detail = callee.getQualifiedName()
  )
}

from string recordType, string kind, string file, int startLine, int startColumn,
  int endLine, int endColumn, string detail, string targetFile, int targetLine,
  int targetColumn, int targetEndLine, int targetEndColumn
where
  exists(DataFlow::Node sink, Location location |
    recordType = "sink" and selectedSink(sink, kind) and location = sink.getLocation() and
    locationInfo(location, file, startLine, startColumn, endLine, endColumn) and
    detail = sink.toString() and
    targetFile = file and targetLine = startLine and targetColumn = startColumn and
    targetEndLine = endLine and targetEndColumn = endColumn
  )
  or
  exists(DataFlow::Node source, DataFlow::Node target, Location sourceLocation, Location targetLocation |
    recordType = "data" and kind = "localFlowStep" and sinkRelevantScope(source.getScope()) and
    DataFlow::localFlowStep(source, target) and
    sourceLocation = source.getLocation() and targetLocation = target.getLocation() and
    locationInfo(sourceLocation, file, startLine, startColumn, endLine, endColumn) and
    locationInfo(targetLocation, targetFile, targetLine, targetColumn, targetEndLine, targetEndColumn) and
    detail = source.toString() + " -> " + target.toString()
  )
  or
  exists(ControlFlowNode source, ControlFlowNode target, Location sourceLocation, Location targetLocation |
    recordType = "control" and kind = "guard" and sinkRelevantScope(source.getScope()) and
    directControlEdge(source, target, detail) and
    sourceLocation = source.getLocation() and targetLocation = target.getLocation() and
    locationInfo(sourceLocation, file, startLine, startColumn, endLine, endColumn) and
    locationInfo(targetLocation, targetFile, targetLine, targetColumn, targetEndLine, targetEndColumn)
  )
  or
  exists(ControlFlowNode source, ControlFlowNode target, Location sourceLocation, Location targetLocation |
    recordType = "exception" and kind = "exceptionalSuccessor" and sinkRelevantScope(source.getScope()) and
    target = source.getAnExceptionalSuccessor() and
    sourceLocation = source.getLocation() and targetLocation = target.getLocation() and
    locationInfo(sourceLocation, file, startLine, startColumn, endLine, endColumn) and
    locationInfo(targetLocation, targetFile, targetLine, targetColumn, targetEndLine, targetEndColumn) and
    detail = source.toString() + " -> " + target.toString()
  )
  or
  exists(ControlFlowNode source, ControlFlowNode target, Location sourceLocation, Location targetLocation |
    recordType = "call" and kind = "dynamicDispatch" and sinkRelevantScope(target.getScope()) and
    directCallEdge(source, target, detail) and
    sourceLocation = source.getLocation() and targetLocation = target.getLocation() and
    locationInfo(sourceLocation, file, startLine, startColumn, endLine, endColumn) and
    locationInfo(targetLocation, targetFile, targetLine, targetColumn, targetEndLine, targetEndColumn)
  )
select recordType, kind, file, startLine, startColumn, endLine, endColumn, detail,
  targetFile, targetLine, targetColumn, targetEndLine, targetEndColumn

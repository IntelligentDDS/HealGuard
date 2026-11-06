/**
 * @name Healing semantic graph attachments
 * @description Exports direct semantic edges whose source is injected healing code.
 * @kind table
 * @id agenticheal/python-healing-impact
 */
import python
import semmle.python.dataflow.new.DataFlow
import semmle.python.GuardedControlFlow
import semmle.python.types.FunctionObject

external predicate healingSource(
  string file, int startLine, int startColumn, int endLine, int endColumn
);

external predicate candidateSink(
  string file, int startLine, int startColumn, int endLine, int endColumn
);

predicate locationInfo(Location location, string file, int startLine, int startColumn, int endLine, int endColumn) {
  location.hasLocationInfo(file, startLine, startColumn, endLine, endColumn)
}

predicate insideHealingSource(Location location) {
  exists(string file, int startLine, int startColumn, int endLine, int endColumn,
    int sourceStartLine, int sourceStartColumn, int sourceEndLine, int sourceEndColumn |
    locationInfo(location, file, startLine, startColumn, endLine, endColumn) and
    healingSource(file, sourceStartLine, sourceStartColumn, sourceEndLine, sourceEndColumn) and
    sourceStartLine <= startLine and endLine <= sourceEndLine and
    (sourceStartLine < startLine or sourceStartColumn <= startColumn) and
    (endLine < sourceEndLine or endColumn <= sourceEndColumn)
  )
}

predicate sinkRelevantScope(Scope scope) {
  exists(AstNode sink, Location location, string file,
    int startLine, int startColumn, int endLine, int endColumn,
    int sinkStartLine, int sinkStartColumn, int sinkEndLine, int sinkEndColumn |
    scope = sink.getScope() and location = sink.getLocation() and
    locationInfo(location, file, startLine, startColumn, endLine, endColumn) and
    candidateSink(file, sinkStartLine, sinkStartColumn, sinkEndLine, sinkEndColumn) and
    sinkStartLine <= startLine and endLine <= sinkEndLine and
    (sinkStartLine < startLine or sinkStartColumn <= startColumn) and
    (endLine < sinkEndLine or endColumn <= sinkEndColumn)
  )
  or
  exists(FunctionObject callee |
    callee.getACall().getScope() = scope and sinkRelevantScope(callee.getFunction())
  )
}

predicate directControlEdge(ControlFlowNode source, ControlFlowNode target, string detail) {
  exists(ConditionBlock condition, boolean branch |
    source = condition.getLastNode() and insideHealingSource(source.getLocation()) and
    condition.controls(target.getBasicBlock(), branch) and
    (branch = true and detail = "true" or branch = false and detail = "false")
  )
}

predicate directCallEdge(ControlFlowNode source, ControlFlowNode target, string detail) {
  exists(FunctionObject callee |
    source = callee.getACall() and insideHealingSource(source.getLocation()) and
    target = callee.getFunction().getEntryNode().getASuccessor() and
    detail = callee.getQualifiedName()
  )
}

from string recordType, string kind, string file, int startLine, int startColumn,
  int endLine, int endColumn, string detail, string targetFile, int targetLine,
  int targetColumn, int targetEndLine, int targetEndColumn
where
  exists(DataFlow::Node source, DataFlow::Node target, Location sourceLocation, Location targetLocation |
    recordType = "data" and kind = "localFlowStep" and sinkRelevantScope(source.getScope()) and
    DataFlow::localFlowStep(source, target) and insideHealingSource(source.getLocation()) and
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
    recordType = "exception" and kind = "exceptionalSuccessor" and
    sinkRelevantScope(source.getScope()) and target = source.getAnExceptionalSuccessor() and
    insideHealingSource(source.getLocation()) and
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

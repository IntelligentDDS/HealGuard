/**
 * @name Sensitive sinks and summaries
 * @description Exports CodeQL Python security sinks and summary models.
 * @kind table
 * @id agenticheal/python-sensitive-sinks
 */
import python
import meta.alerts.Sinks
import semmle.python.frameworks.data.ModelsAsData
import semmle.python.dataflow.new.DataFlow

predicate locationInfo(Location location, string file, int startLine, int startColumn, int endLine, int endColumn) {
  location.hasLocationInfo(file, startLine, startColumn, endLine, endColumn)
}

from string recordType, string kind, string file, int startLine, int startColumn,
  int endLine, int endColumn, string detail, string sinkFile, int sinkLine,
  int sinkColumn, int sinkEndLine, int sinkEndColumn
where
  exists(DataFlow::Node sink, Location location |
    recordType = "sink" and (sink = taintSink(kind) or ModelOutput::sinkNode(sink, kind)) and
    location = sink.getLocation() and
    locationInfo(location, file, startLine, startColumn, endLine, endColumn) and
    detail = sink.toString() and sinkFile = file and sinkLine = startLine and
    sinkColumn = startColumn and sinkEndLine = endLine and sinkEndColumn = endColumn
  )
  or
  exists(string type, string path, string input, string output, string model |
    recordType = "summary" and ModelOutput::relevantSummaryModel(type, path, input, output, kind, model) and
    file = type and startLine = 0 and startColumn = 0 and endLine = 0 and endColumn = 0 and
    detail = path + ";" + input + ";" + output + ";" + model and
    sinkFile = "" and sinkLine = 0 and sinkColumn = 0 and sinkEndLine = 0 and sinkEndColumn = 0
  )
select recordType, kind, file, startLine, startColumn, endLine, endColumn, detail,
  sinkFile, sinkLine, sinkColumn, sinkEndLine, sinkEndColumn

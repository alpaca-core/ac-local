# Copyright (c) Alpaca Core
# SPDX-License-Identifier: MIT
#

# on Windows doxygen refuses to generate multiple subdirectories (as mkdir -p)
# so we need to prepare the root doc directory manually
OUT_DOC_DIR = 'out/doc'
directory OUT_DOC_DIR

task generate_cpp_doc: OUT_DOC_DIR do
  sh 'doxygen'
end

task generate_c_doc: OUT_DOC_DIR do
  sh 'doxygen wrapper/c/Doxyfile.c-api'
end

task generate_java_doc: OUT_DOC_DIR do
  sh 'doxygen wrapper/java/Doxyfile.java-api'
end

task generate_swift_doc: OUT_DOC_DIR do
  docc = ''
  docc = 'xcrun ' if RUBY_PLATFORM =~ /darwin/
  docc += 'docc'

  sh %W(
    #{docc} convert
    --emit-lmdb-index
    --fallback-display-name AlpacaCoreSwift
    --fallback-bundle-identifier AlpacaCoreSwift
    --fallback-bundle-version 0
    --output-dir #{OUT_DOC_DIR}/swift/AlpacaCoreSwift.doccarchive
    --diagnostics-file #{OUT_DOC_DIR}/swift/AlpacaCoreSwift-diagnostics.json
    --additional-symbol-graph-dir #{OUT_DOC_DIR}/swift/symbol-graph
  ).join(' ')
end

desc 'Generate all documentation'
task generate_doc: %i[generate_cpp_doc generate_c_doc generate_java_doc generate_swift_doc]

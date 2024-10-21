# Copyright (c) Alpaca Core
# SPDX-License-Identifier: MIT
#
require 'nokogiri'
require_relative 'fixygen'

XML_DIR = File.expand_path(Dir.pwd + '/../../out/doc/c-api/xml')

# find groups in index.xml
groups = Nokogiri::XML(File.open(XML_DIR + '/index.xml')).root.elements.select {
  _1[:kind] == 'group'
}.map { |g|
  name = g.elements.find { _1.name == 'name' }.inner_text
  [name, g[:refid]]
}.to_h

p groups

"""Tests for server-side tools."""

from __future__ import annotations

from ai_arch_toolkit.core._server_tools import ServerTool, code_execution, web_search
from ai_arch_toolkit.core._tools import prepare_tools


class TestServerTool:
    def test_web_search_helper(self):
        tool = web_search()
        assert isinstance(tool, ServerTool)
        assert tool.type == "web_search"
        assert tool.config == {}

    def test_code_execution_helper(self):
        tool = code_execution()
        assert isinstance(tool, ServerTool)
        assert tool.type == "code_execution"

    def test_web_search_with_config(self):
        tool = web_search(max_results=5)
        assert tool.config == {"max_results": 5}


class TestPrepareToolsServerTool:
    def test_server_tool_in_list(self):
        tools = prepare_tools([web_search()])
        assert tools is not None
        assert len(tools) == 1
        assert tools[0]["_server_tool"] is True
        assert tools[0]["type"] == "web_search"

    def test_mixed_server_and_function_tools(self):
        fn_tool = {"name": "get_weather", "input_schema": {}}
        tools = prepare_tools([fn_tool, web_search()])
        assert tools is not None
        assert len(tools) == 2
        assert not tools[0].get("_server_tool")
        assert tools[1]["_server_tool"] is True

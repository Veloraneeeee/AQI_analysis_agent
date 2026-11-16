"""Offline regression checks: no historical datasets or Ollama service required."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import aqi_analysis as analysis
import data_sources as sources
import main
import uploaded_file_analysis as uploaded
from langgraph.types import Command
from types import SimpleNamespace


class RefactorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.events = Path(self.temp.name) / 'events.xlsx'
        self.aqi = Path(self.temp.name) / 'aqi.csv'
        pd.DataFrame([
            [' Flash Flood ', '2024-01-04', '2024-01-05', 48, 1, 'C'],
            ['Flood', '2023-01-04', '2023-01-05', 48, 1, 'C'],
            ['Wind', '2024-01-04', '2024-01-05', 6, 1, 'C'],
            ['Flash Flood', '2024-01-04', '2024-01-05', 48, 1, 'Z'],
        ], columns=['EVENT_TYPE', 'BEGIN_DATE_TIME', 'END_DATE_TIME',
                    'STATE_FIPS', 'CZ_FIPS', 'CZ_TYPE']).to_excel(self.events, index=False)
        # Same-day duplicates use a median; a different county must never match.
        pd.DataFrame([
            ['2024-01-04', 10, 48, 1, 1, 1],
            ['2024-01-04', 30, 48, 1, 1, 1],
            ['2024-01-05', 25, 48, 1, 1, 1],
            ['2024-01-04', 999, 48, 2, 1, 1],
        ], columns=['Date Local', 'AQI', 'State Code', 'County Code',
                    'Site Num', 'POC']).to_csv(self.aqi, index=False)

    def test_discovery_tool_contract(self):
        with patch.object(sources, 'EVENT_FILE_PATH', self.events):
            result = main.query_weather_events.invoke({'region': ' TX ', 'year': 2024})
        self.assertEqual(result, {
            'region': ' TX ', 'state_fips': '48', 'year': 2024,
            'year_basis': '事件开始日期所在年份', 'matched_row_count': 2,
            'event_types': ['Flash Flood'], 'event_type_count': 1,
        })

    def test_discovery_empty_and_validation(self):
        with patch.object(sources, 'EVENT_FILE_PATH', self.events):
            self.assertEqual(sources.get_weather_event_types('得州', 2025)['matched_row_count'], 0)
            with self.assertRaises(ValueError):
                sources.get_weather_event_types('得州', 0)
            with self.assertRaises(ValueError):
                sources.get_weather_event_types('unknown', 2024)
            events = pd.read_excel(self.events)
            events.loc[0, 'BEGIN_DATE_TIME'] = 'bad date'
            events.to_excel(self.events, index=False)
            with self.assertRaisesRegex(ValueError, '1条记录'):
                sources.get_weather_event_types('得州', 2024)

    def test_uploaded_analysis_node_keeps_endpoint_semantics(self):
        result = uploaded.match_and_calculate({
            'event_file_path': str(self.events), 'aqi_file_path': str(self.aqi),
            'column_mapping': sources.COLUMN_MAPPING,
            'selected_event_type': 'Flash Flood',
        })
        self.assertEqual(result['aqi_changes'], [{
            'event_id': 0, 'event_type': 'Flash Flood', 'state_code': '48',
            'county_code': '001', 'monitor_site_id': '48|1|1|1',
            'start_date': '2024-01-04', 'end_date': '2024-01-05',
            'start_aqi': 20.0, 'end_aqi': 25.0, 'aqi_change': 5.0,
            'trend': 'increased',
        }])
        summary = uploaded.summarize_results({**result, 'selected_event_type': 'Flash Flood'})
        self.assertEqual(summary['trend_summary']['median_aqi_change'], 5.0)

    def test_uploaded_missing_end_is_an_error(self):
        pd.read_csv(self.aqi).iloc[[0]].to_csv(self.aqi, index=False)
        with self.assertRaisesRegex(ValueError, '完整 AQI'):
            analysis.analyze_event_aqi(str(self.events), str(self.aqi), sources.COLUMN_MAPPING, 'Flash Flood')

    def test_windows_still_exclude_event_days_and_require_completeness(self):
        windows = analysis.build_event_windows('2024-01-04', '2024-01-05', 1, 1)
        data = pd.DataFrame({'Date Local': ['2024-01-03', '2024-01-04', '2024-01-06'],
                             'AQI': [10, 999, 30]})
        self.assertEqual(analysis.compare_monitor_windows(data, windows)['aqi_change'], 20)
        self.assertIsNone(analysis.compare_monitor_windows(data.iloc[:2], windows)['aqi_change'])

    def test_analysis_tool_delegates_and_preserves_status(self):
        for count, status in [(0, 'no_complete_comparison'), (1, 'ok')]:
            with self.subTest(status=status), patch.object(main, 'analyze_event_windows', return_value={
                'rows_with_complete_comparison': count, 'event_results': ['details'],
            }) as call:
                result = main.analyze_weather_aqi.invoke({'region': '得州', 'year': 2024, 'event_type': 'Flood'})
                call.assert_called_once_with(region='得州', year=2024, event_type='Flood', before_days=3, after_days=3)
                self.assertEqual(result['status'], status)
                self.assertNotIn('event_results', result)

    def test_graph_pauses_resumes_and_completes(self):
        config = {'configurable': {'thread_id': 'refactor-regression'}}
        state = {
            'event_file_path': str(self.events), 'aqi_file_path': str(self.aqi),
            'column_mapping': sources.COLUMN_MAPPING,
        }
        from unittest.mock import Mock
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(content='fixture report')
        graph = uploaded.build_uploaded_file_analysis_graph(main.AnalysisState, llm)
        paused = graph.invoke(state, config)
        self.assertTrue(paused['__interrupt__'])
        completed = graph.invoke(Command(resume='Flash Flood'), config)
        self.assertEqual(completed['final_report'], 'fixture report')
        self.assertEqual(completed['trend_summary']['median_aqi_change'], 5.0)

    def test_shared_event_type_tool_still_serves_chat(self):
        with patch.object(main, 'EVENT_FILE_PATH', self.events):
            self.assertEqual(main.list_weather_events.invoke({}), ['Flash Flood', 'Flood', 'Wind'])
        self.assertIs(main.extract_types, uploaded.extract_types)

    def test_mapping_model_is_injected_without_importing_main(self):
        from unittest.mock import Mock
        llm = Mock()
        llm.with_structured_output.return_value.invoke.return_value = uploaded.ColumnMapping(**sources.COLUMN_MAPPING)
        state = {
            'event_file_path': str(self.events), 'aqi_file_path': str(self.aqi),
            'event_file_preview': 'events', 'aqi_file_preview': 'aqi',
            'column_mapping': sources.COLUMN_MAPPING, 'confirmation_feedback': 'correct column',
        }
        self.assertEqual(uploaded.identity_columns(state, llm=llm)['column_mapping'], sources.COLUMN_MAPPING)
        self.assertFalse(uploaded.revise_columns(state, llm=llm)['user_confirmed'])
        llm.with_structured_output.assert_called_with(uploaded.ColumnMapping)

    def test_confirmation_interrupt_and_both_routes(self):
        from langgraph.graph import StateGraph, START, END
        from langgraph.checkpoint.memory import InMemorySaver
        for reply, expected in [('正确', 'extract_event_types'), ('修改县列', 'revise_columns')]:
            with self.subTest(reply=reply):
                builder = StateGraph(main.AnalysisState)
                builder.add_node('confirm', uploaded.ask_for_confirmation)
                builder.add_node('extract_event_types', lambda state: {'final_report': 'extract_event_types'})
                builder.add_node('revise_columns', lambda state: {'final_report': 'revise_columns'})
                builder.add_edge(START, 'confirm')
                builder.add_conditional_edges('confirm', uploaded.route_confirmation)
                builder.add_edge('extract_event_types', END)
                builder.add_edge('revise_columns', END)
                graph = builder.compile(checkpointer=InMemorySaver())
                config = {'configurable': {'thread_id': reply}}
                self.assertTrue(graph.invoke({'column_mapping': sources.COLUMN_MAPPING}, config)['__interrupt__'])
                self.assertEqual(graph.invoke(Command(resume=reply), config)['final_report'], expected)

    def test_graphs_and_tool_interfaces_still_exist(self):
        self.assertIn('match_and_calculate', main.analysis_graph.nodes)
        self.assertIn('summarize_results', main.analysis_graph.nodes)
        self.assertIn('chat_with_user', main.graph.nodes)
        self.assertEqual(main.query_weather_events.name, 'query_weather_events')
        self.assertEqual(set(main.query_weather_events.args), {'region', 'year'})
        self.assertEqual(set(main.analyze_weather_aqi.args),
                         {'region', 'year', 'event_type', 'before_days', 'after_days'})


if __name__ == '__main__':
    unittest.main()

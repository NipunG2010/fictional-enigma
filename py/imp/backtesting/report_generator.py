"""
Comprehensive backtest report generation with charts and statistics.

This module implements the ReportGenerator class that creates detailed
backtest reports in multiple formats (JSON, CSV, HTML) with visualizations
and comprehensive statistics.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available, HTML reports will not include charts")


logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate comprehensive backtest reports with charts and statistics.
    
    This class creates standardized report outputs compatible with analysis tools,
    including detailed performance metrics, trade analysis, and visualizations.
    
    Requirements: 1.5, 4.1, 4.2
    """
    
    def __init__(self):
        """Initialize ReportGenerator."""
        self.report_data = {}
    
    def generate_json_report(
        self,
        results: Any,  # BacktestResults
        output_path: Path
    ) -> None:
        """
        Generate JSON format report.
        
        Args:
            results: BacktestResults object
            output_path: Path to save JSON report
            
        Requirements: 1.5
        """
        logger.info(f"Generating JSON report: {output_path}")
        
        report = {
            'metadata': {
                'report_generated': datetime.now().isoformat(),
                'backtest_name': results.config.name,
                'description': results.config.description,
                'execution_time': results.execution_time,
            },
            'configuration': results.config.model_dump(mode='json'),
            'performance': results.performance_metrics.to_dict(),
            'portfolio': {
                'initial_capital': results.config.initial_capital,
                'final_value': results.final_portfolio_value,
                'total_return': results.performance_metrics.total_return,
                'total_pnl': results.final_portfolio_value - results.config.initial_capital,
            },
            'trades': {
                'total_trades': len(results.orders),
                'win_rate': results.performance_metrics.win_rate,
                'profit_factor': results.performance_metrics.profit_factor,
                'avg_win': results.performance_metrics.avg_win,
                'avg_loss': results.performance_metrics.avg_loss,
            },
            'risk_metrics': {
                'sharpe_ratio': results.performance_metrics.sharpe_ratio,
                'sortino_ratio': results.performance_metrics.sortino_ratio,
                'calmar_ratio': results.performance_metrics.calmar_ratio,
                'max_drawdown': results.performance_metrics.max_drawdown,
                'volatility': results.performance_metrics.annualized_volatility,
                'var_95': results.performance_metrics.var_95,
                'cvar_95': results.performance_metrics.cvar_95,
            },
            'data_quality': {
                'quality_score': results.data_quality_report.quality_score if results.data_quality_report else None,
                'total_records': results.data_quality_report.total_records if results.data_quality_report else 0,
                'warnings': results.data_quality_report.warnings if results.data_quality_report else [],
                'errors': results.data_quality_report.errors if results.data_quality_report else [],
            }
        }
        
        # Add regime metrics if available
        if results.regime_metrics:
            report['regime_analysis'] = results.regime_metrics
        
        # Add validation report if available
        if results.validation_report:
            report['walk_forward_validation'] = results.validation_report.to_dict()
        
        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"JSON report saved to {output_path}")
    
    def generate_csv_reports(
        self,
        results: Any,  # BacktestResults
        output_dir: Path
    ) -> Dict[str, Path]:
        """
        Generate CSV format reports (trades, portfolio, metrics).
        
        Args:
            results: BacktestResults object
            output_dir: Directory to save CSV reports
            
        Returns:
            Dictionary mapping report type to file path
            
        Requirements: 1.5
        """
        logger.info(f"Generating CSV reports in {output_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = {}
        
        # Trades CSV
        if results.orders:
            trades_path = output_dir / f"{results.config.name}_trades.csv"
            self._generate_trades_csv(results.orders, trades_path)
            saved_files['trades'] = trades_path
        
        # Portfolio snapshots CSV
        if results.portfolio_snapshots:
            portfolio_path = output_dir / f"{results.config.name}_portfolio.csv"
            self._generate_portfolio_csv(results.portfolio_snapshots, portfolio_path)
            saved_files['portfolio'] = portfolio_path
        
        # Performance metrics CSV
        metrics_path = output_dir / f"{results.config.name}_metrics.csv"
        self._generate_metrics_csv(results.performance_metrics, metrics_path)
        saved_files['metrics'] = metrics_path
        
        # Trade costs CSV
        if results.trade_costs:
            costs_path = output_dir / f"{results.config.name}_costs.csv"
            self._generate_costs_csv(results.trade_costs, costs_path)
            saved_files['costs'] = costs_path
        
        logger.info(f"Generated {len(saved_files)} CSV reports")
        return saved_files
    
    def generate_html_report(
        self,
        results: Any,  # BacktestResults
        output_path: Path,
        include_charts: bool = True
    ) -> None:
        """
        Generate HTML format report with charts and statistics.
        
        Args:
            results: BacktestResults object
            output_path: Path to save HTML report
            include_charts: Whether to include interactive charts
            
        Requirements: 1.5, 4.1, 4.2
        """
        logger.info(f"Generating HTML report: {output_path}")
        
        if include_charts and not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available, charts will be omitted")
            include_charts = False
        
        # Generate HTML content
        html_content = self._generate_html_content(results, include_charts)
        
        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {output_path}")
    
    def _generate_trades_csv(self, orders: List[Any], output_path: Path) -> None:
        """Generate trades CSV file."""
        trades_data = []
        
        for order in orders:
            trades_data.append({
                'order_id': order.order_id,
                'timestamp': order.timestamp.isoformat(),
                'symbol': order.symbol,
                'side': order.side,
                'quantity': order.quantity,
                'price': order.price,
                'notional_value': order.notional_value,
                'order_type': order.order_type,
                'position_size_method': order.position_size_method,
                'signal_source': order.metadata.get('signal_source', 'unknown'),
                'signal_confidence': order.metadata.get('signal_confidence', 0.0),
                'regime_state': order.metadata.get('regime_state'),
            })
        
        df = pd.DataFrame(trades_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(trades_data)} trades to {output_path}")
    
    def _generate_portfolio_csv(self, snapshots: List[Any], output_path: Path) -> None:
        """Generate portfolio snapshots CSV file."""
        portfolio_data = []
        
        for snapshot in snapshots:
            portfolio_data.append({
                'timestamp': snapshot.timestamp.isoformat(),
                'cash': snapshot.cash,
                'market_value': snapshot.market_value,
                'total_value': snapshot.total_value,
                'unrealized_pnl': snapshot.unrealized_pnl,
                'realized_pnl': snapshot.realized_pnl,
                'total_pnl': snapshot.total_pnl,
                'num_positions': snapshot.num_positions,
                'leverage': snapshot.leverage,
            })
        
        df = pd.DataFrame(portfolio_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(portfolio_data)} portfolio snapshots to {output_path}")
    
    def _generate_metrics_csv(self, metrics: Any, output_path: Path) -> None:
        """Generate performance metrics CSV file."""
        metrics_dict = metrics.to_dict()
        
        # Convert to DataFrame with metric name and value columns
        metrics_data = [
            {'metric': key, 'value': value}
            for key, value in metrics_dict.items()
            if not isinstance(value, (dict, list))
        ]
        
        df = pd.DataFrame(metrics_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved performance metrics to {output_path}")
    
    def _generate_costs_csv(self, trade_costs: List[Any], output_path: Path) -> None:
        """Generate trade costs CSV file."""
        costs_data = [cost.to_dict() for cost in trade_costs]
        
        df = pd.DataFrame(costs_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(costs_data)} trade costs to {output_path}")
    
    def _generate_html_content(
        self,
        results: Any,
        include_charts: bool
    ) -> str:
        """Generate HTML report content."""
        # HTML header
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report - {name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
        }}
        .metric-label {{
            font-size: 14px;
            color: #7f8c8d;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-value.positive {{
            color: #27ae60;
        }}
        .metric-value.negative {{
            color: #e74c3c;
        }}
        .info-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .info-table th, .info-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        .info-table th {{
            background-color: #34495e;
            color: white;
            font-weight: 600;
        }}
        .info-table tr:hover {{
            background-color: #f8f9fa;
        }}
        .warning {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
        .error {{
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
        .success {{
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
        .chart-container {{
            margin: 30px 0;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Backtest Report: {name}</h1>
        <p><strong>Generated:</strong> {generated_time}</p>
        <p><strong>Description:</strong> {description}</p>
        
""".format(
            name=results.config.name,
            generated_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            description=results.config.description or 'N/A'
        )
        
        # Summary metrics
        html += self._generate_summary_section(results)
        
        # Performance metrics
        html += self._generate_performance_section(results)
        
        # Risk metrics
        html += self._generate_risk_section(results)
        
        # Trade statistics
        html += self._generate_trade_statistics_section(results)
        
        # Data quality
        if results.data_quality_report:
            html += self._generate_data_quality_section(results.data_quality_report)
        
        # Charts
        if include_charts and PLOTLY_AVAILABLE:
            html += self._generate_charts_section(results)
        
        # Configuration
        html += self._generate_configuration_section(results.config)
        
        # Footer
        html += """
        <div class="footer">
            <p>Generated by IMP Backtesting Framework</p>
            <p>Execution time: {:.2f} seconds</p>
        </div>
    </div>
</body>
</html>
""".format(results.execution_time)
        
        return html
    
    def _generate_summary_section(self, results: Any) -> str:
        """Generate summary metrics section."""
        metrics = results.performance_metrics
        
        total_return_class = 'positive' if metrics.total_return > 0 else 'negative'
        sharpe_class = 'positive' if metrics.sharpe_ratio > 1.0 else 'negative' if metrics.sharpe_ratio < 0 else ''
        
        return f"""
        <h2>Summary</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Initial Capital</div>
                <div class="metric-value">${results.config.initial_capital:,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Final Value</div>
                <div class="metric-value">${results.final_portfolio_value:,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {total_return_class}">{metrics.total_return:.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value {sharpe_class}">{metrics.sharpe_ratio:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative">{metrics.max_drawdown:.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Number of Trades</div>
                <div class="metric-value">{metrics.num_trades}</div>
            </div>
        </div>
"""
    
    def _generate_performance_section(self, results: Any) -> str:
        """Generate performance metrics section."""
        metrics = results.performance_metrics
        
        return f"""
        <h2>Performance Metrics</h2>
        <table class="info-table">
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Return</td>
                <td>{metrics.total_return:.2%}</td>
            </tr>
            <tr>
                <td>Annualized Return</td>
                <td>{metrics.annualized_return:.2%}</td>
            </tr>
            <tr>
                <td>Cumulative Return</td>
                <td>{metrics.cumulative_return:.2%}</td>
            </tr>
            <tr>
                <td>Volatility (Annualized)</td>
                <td>{metrics.annualized_volatility:.2%}</td>
            </tr>
            <tr>
                <td>Number of Periods</td>
                <td>{metrics.num_periods}</td>
            </tr>
        </table>
"""
    
    def _generate_risk_section(self, results: Any) -> str:
        """Generate risk metrics section."""
        metrics = results.performance_metrics
        
        return f"""
        <h2>Risk Metrics</h2>
        <table class="info-table">
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Sharpe Ratio</td>
                <td>{metrics.sharpe_ratio:.2f}</td>
            </tr>
            <tr>
                <td>Sortino Ratio</td>
                <td>{metrics.sortino_ratio:.2f}</td>
            </tr>
            <tr>
                <td>Calmar Ratio</td>
                <td>{metrics.calmar_ratio:.2f}</td>
            </tr>
            <tr>
                <td>Maximum Drawdown</td>
                <td>{metrics.max_drawdown:.2%}</td>
            </tr>
            <tr>
                <td>Average Drawdown</td>
                <td>{metrics.avg_drawdown:.2%}</td>
            </tr>
            <tr>
                <td>Max Drawdown Duration</td>
                <td>{metrics.max_drawdown_duration} periods</td>
            </tr>
            <tr>
                <td>Value at Risk (95%)</td>
                <td>{metrics.var_95:.2%}</td>
            </tr>
            <tr>
                <td>Conditional VaR (95%)</td>
                <td>{metrics.cvar_95:.2%}</td>
            </tr>
        </table>
"""
    
    def _generate_trade_statistics_section(self, results: Any) -> str:
        """Generate trade statistics section."""
        metrics = results.performance_metrics
        
        return f"""
        <h2>Trade Statistics</h2>
        <table class="info-table">
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Trades</td>
                <td>{metrics.num_trades}</td>
            </tr>
            <tr>
                <td>Win Rate</td>
                <td>{metrics.win_rate:.2%}</td>
            </tr>
            <tr>
                <td>Profit Factor</td>
                <td>{metrics.profit_factor:.2f}</td>
            </tr>
            <tr>
                <td>Average Win</td>
                <td>${metrics.avg_win:.2f}</td>
            </tr>
            <tr>
                <td>Average Loss</td>
                <td>${metrics.avg_loss:.2f}</td>
            </tr>
        </table>
"""
    
    def _generate_data_quality_section(self, report: Any) -> str:
        """Generate data quality section."""
        quality_class = 'success' if report.quality_score >= 0.8 else 'warning' if report.quality_score >= 0.7 else 'error'
        
        html = f"""
        <h2>Data Quality</h2>
        <div class="{quality_class}">
            <strong>Quality Score:</strong> {report.quality_score:.2f} / 1.00
        </div>
        <table class="info-table">
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Records</td>
                <td>{report.total_records:,}</td>
            </tr>
            <tr>
                <td>Symbols</td>
                <td>{len(report.symbols)}</td>
            </tr>
            <tr>
                <td>Date Range</td>
                <td>{report.date_range[0].strftime('%Y-%m-%d')} to {report.date_range[1].strftime('%Y-%m-%d')}</td>
            </tr>
        </table>
"""
        
        if report.warnings:
            html += '<div class="warning"><strong>Warnings:</strong><ul>'
            for warning in report.warnings[:5]:  # Show first 5 warnings
                html += f'<li>{warning}</li>'
            html += '</ul></div>'
        
        if report.errors:
            html += '<div class="error"><strong>Errors:</strong><ul>'
            for error in report.errors[:5]:  # Show first 5 errors
                html += f'<li>{error}</li>'
            html += '</ul></div>'
        
        return html
    
    def _generate_charts_section(self, results: Any) -> str:
        """Generate charts section with Plotly."""
        if not PLOTLY_AVAILABLE:
            return ""
        
        html = '<h2>Charts</h2>'
        
        # Equity curve chart
        equity_chart = self._create_equity_curve_chart(results)
        if equity_chart:
            html += '<div class="chart-container">'
            html += equity_chart
            html += '</div>'
        
        # Drawdown chart
        drawdown_chart = self._create_drawdown_chart(results)
        if drawdown_chart:
            html += '<div class="chart-container">'
            html += drawdown_chart
            html += '</div>'
        
        return html
    
    def _create_equity_curve_chart(self, results: Any) -> Optional[str]:
        """Create equity curve chart."""
        if not results.portfolio_snapshots:
            return None
        
        timestamps = [s.timestamp for s in results.portfolio_snapshots]
        values = [s.total_value for s in results.portfolio_snapshots]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#3498db', width=2)
        ))
        
        fig.update_layout(
            title='Equity Curve',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    def _create_drawdown_chart(self, results: Any) -> Optional[str]:
        """Create drawdown chart."""
        if not results.portfolio_snapshots:
            return None
        
        timestamps = [s.timestamp for s in results.portfolio_snapshots]
        values = [s.total_value for s in results.portfolio_snapshots]
        
        # Calculate drawdown
        running_max = pd.Series(values).cummax()
        drawdown = (pd.Series(values) - running_max) / running_max
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=drawdown * 100,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='#e74c3c', width=2)
        ))
        
        fig.update_layout(
            title='Drawdown',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    def _generate_configuration_section(self, config: Any) -> str:
        """Generate configuration section."""
        return f"""
        <h2>Configuration</h2>
        <table class="info-table">
            <tr>
                <th>Parameter</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Date Range</td>
                <td>{config.start_date} to {config.end_date}</td>
            </tr>
            <tr>
                <td>Symbols</td>
                <td>{', '.join(config.symbols)}</td>
            </tr>
            <tr>
                <td>Initial Capital</td>
                <td>${config.initial_capital:,.2f}</td>
            </tr>
            <tr>
                <td>Position Sizing Method</td>
                <td>{config.position_sizing.method.value}</td>
            </tr>
            <tr>
                <td>Max Position Size</td>
                <td>{config.position_sizing.max_position_size:.1%}</td>
            </tr>
            <tr>
                <td>Asset Class</td>
                <td>{config.cost_model.asset_class.value}</td>
            </tr>
            <tr>
                <td>Commission Rate</td>
                <td>{config.cost_model.commission_rate:.3%}</td>
            </tr>
        </table>
"""

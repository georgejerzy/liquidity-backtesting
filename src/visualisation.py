from typing import Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd



def export


def visualise_strategy(
    df_positions: pd.DataFrame,
    actuator_alt: Any,
    actuator_np: Any,
    market_key: str,
    width: int = None,
    height: int = 1000,
) -> go.Figure:
    """
    Create a Plotly figure showing ETH price with an 8h rolling average,
    position rectangles with a midpoint red line, and a net value comparison.

    Parameters:
    - df_positions: DataFrame containing position data with columns:
      'open_time', 'close_time', 'lower_quote_price', 'upper_quote_price', etc.
    - actuator_alt: Object containing account_status_df and strategy.data for the alternative strategy.
    - actuator_np: Object containing account_status_df for the no-provision strategy.
    - market_key: Key for accessing market-specific data from actuator_alt.strategy.data.
    - width: Width of the resulting figure in pixels (default: 800).
    - height: Height of the resulting figure in pixels (default: 800).

    Returns:
    - fig: A Plotly Figure object.
    """

    # Extract series from the actuators.
    eth_price_series = actuator_alt.account_status_df["price"]["ETH"]
    rolling_avg_series = actuator_alt.strategy.data[market_key]["rolling_avg"]
    net_value_series_alt = actuator_alt.account_status_df["net_value"]
    net_value_series_np = actuator_np.account_status_df["net_value"]

    # Create a figure with 2 rows (top: price & rolling average, bottom: net value).
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            "ETH Price with 8h Rolling Average & Positions",
            "Net Value Comparison",
        ),
    )

    # Top subplot: ETH Price and Rolling Average.
    fig.add_trace(
        go.Scatter(
            x=eth_price_series.index, y=eth_price_series, mode="lines", name="ETH Price"
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=rolling_avg_series.index,
            y=rolling_avg_series,
            mode="lines",
            name="8h Rolling Average",
            line=dict(dash="dash", color="gray"),
        ),
        row=1,
        col=1,
    )

    # Build a list of shape definitions for positions.
    shapes = []
    for _, pos in df_positions.iterrows():
        # Only add shapes if the position has a valid close_time.
        if pd.notnull(pos["close_time"]):
            # Blue rectangle for the position's price range.
            rect = dict(
                type="rect",
                xref="x",
                yref="y",
                x0=pos["open_time"],
                x1=pos["close_time"],
                y0=pos["lower_quote_price"],
                y1=pos["upper_quote_price"],
                fillcolor="blue",
                opacity=0.3,
                line_width=0,
                layer="below",
            )
            shapes.append(rect)

            # Calculate midpoint between lower and upper quote prices.
            mid_val = (pos["lower_quote_price"] + pos["upper_quote_price"]) / 2
            # Red line at the midpoint.
            line = dict(
                type="line",
                xref="x",
                yref="y",
                x0=pos["open_time"],
                x1=pos["close_time"],
                y0=mid_val,
                y1=mid_val,
                line=dict(color="red", width=1),
                layer="above",
            )
            shapes.append(line)

    # Bottom subplot: Net Value Comparison.
    fig.add_trace(
        go.Scatter(
            x=net_value_series_alt.index,
            y=net_value_series_alt,
            mode="lines",
            name="Alternative Strategy",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=net_value_series_np.index,
            y=net_value_series_np,
            mode="lines",
            name="No Provision Strategy",
        ),
        row=2,
        col=1,
    )

    # Update layout settings, set figure dimensions to a square (1:1 ratio),
    # and add all shapes at once.
    fig.update_layout(
        title="ETH Price (with Rolling Average & Positions) and Net Value Comparison",
        xaxis2_title="Date",
        template="plotly_white",
        width=width,
        height=height,
        shapes=shapes,
    )
    fig.update_yaxes(title_text="ETH Price (USDC)", row=1, col=1)
    fig.update_yaxes(title_text="Net Value (USDC)", row=2, col=1)

    return fig

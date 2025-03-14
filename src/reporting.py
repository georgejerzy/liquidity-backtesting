from typing import Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from demeter import Actuator, BaseAction

import pandas as pd
from dataclasses import asdict


from decimal import Decimal


import os
import json
from datetime import datetime
from pathlib import Path


import json
from typing import List

from pydantic import BaseModel


class StrategyResults(BaseModel):
    diff_from_baseline_at_end: float
    percent_increase_from_baseline_at_end: float


from pydantic import BaseModel


class StrategyResults(BaseModel):
    diff_from_baseline_at_end: float
    percent_increase_from_baseline_at_end: float


def calculate_strategy_results(actuator_rolling, actuator_baseline) -> StrategyResults:
    # Extract net value series from each actuator's account_status_df
    net_value_series_alt = actuator_rolling.account_status_df["net_value"]
    net_value_series_np = actuator_baseline.account_status_df["net_value"]

    # Calculate the difference and percent increase at the end using .iloc for positional indexing
    diff = net_value_series_alt.iloc[-1] - net_value_series_np.iloc[-1]
    percent_increase = diff / net_value_series_np.iloc[-1]

    # Return the results wrapped in the StrategyResults model
    return StrategyResults(
        diff_from_baseline_at_end=diff,
        percent_increase_from_baseline_at_end=percent_increase,
    )


def serialize_actions_to_jsonl(actions: List[BaseAction], file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        for action in actions:
            ad = asdict(action)
            ad["market"] = ad["market"].name
            ad["action_type"] = ad["action_type"].name
            ad["timestamp"] = ad["timestamp"].isoformat()
            for k, v in ad.items():
                if isinstance(v, Decimal):
                    ad[k] = float(v)
            json.dump(ad, f, ensure_ascii=False)
            f.write("\n")


def get_fees_dataframe(actuator, market_key):
    fees_payed = []

    # Retrieve the market from the broker using the market_key.
    market = actuator.broker.markets[market_key]

    # Get the ETH price for conversion.
    price_data = market.get_price_from_data()
    price_df = price_data[0] if isinstance(price_data, tuple) else price_data
    eth_price_value = price_df.iloc[-1]["ETH"]
    eth_price = Decimal(str(eth_price_value))

    # Loop over actions and process only uni_lp_buy and uni_lp_sell actions.
    for action in actuator._action_list:
        if action.action_type.name not in ["uni_lp_buy", "uni_lp_sell"]:
            continue

        timestamp = action.timestamp
        action_type = action.action_type.name
        fee_value = Decimal(action.fee)  # fee field as string/Decimal

        # For uni_lp_buy: fee is in the quote token (USDC)
        # For uni_lp_sell: fee is in the base token (ETH)
        if action_type == "uni_lp_buy":
            fee_token0 = fee_value  # fee in token0 (USDC) if _is_token0_quote is True.
            fee_token1 = Decimal("0")
        elif action_type == "uni_lp_sell":
            fee_token0 = Decimal("0")
            fee_token1 = fee_value  # fee in token1 (ETH)

        # Convert using the market's _convert_pair method.
        fee_base, fee_quote = market._convert_pair(fee_token0, fee_token1)
        fee_usd = fee_quote + (fee_base * eth_price)

        fees_payed.append(
            {"timestamp": timestamp, "fee_usd": fee_usd, "action_type": action_type}
        )

    df = pd.DataFrame(fees_payed)
    # Convert the timestamp column to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df["cum_fee_usd"] = df["fee_usd"].cumsum()
    # Optionally, set timestamp as index:
    df.set_index("timestamp", inplace=True)
    return df


def get_positions_dataframe(actions):
    positions = []
    current_position = None

    for action in actions:
        # Convert dataclass instance to dict.
        ad = asdict(action)
        # Convert the enum to its string representation.
        action_type = str(ad.get("action_type"))

        if action_type == "uni_lp_add_liquidity":
            # Start a new position.
            current_position = {
                "open_time": ad.get("timestamp"),
                "lower_tick": ad.get("position")[0],
                "upper_tick": ad.get("position")[1],
                "lower_quote_price": ad.get("lower_quote_price"),
                "upper_quote_price": ad.get("upper_quote_price"),
                "liquidity": ad.get("liquidity"),
                "base_amount": None,
                "quote_amount": None,
                "close_time": None,
            }
        elif action_type == "uni_lp_remove_liquidity":
            if current_position:
                # Close the current position.
                current_position["close_time"] = ad.get("timestamp")
                current_position["base_amount"] = ad.get("base_amount")
                current_position["quote_amount"] = ad.get("quote_amount")
                positions.append(current_position)
                current_position = None  # Reset current position

    return pd.DataFrame(positions)


import pandas as pd
from decimal import Decimal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from decimal import Decimal
import plotly.graph_objects as go
from plotly.subplots import make_subplots


import pandas as pd
from decimal import Decimal
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualise_strategy(
    df_positions: pd.DataFrame,
    actuator_alt: any,
    actuator_np: any,
    df_fees: pd.DataFrame,  # DataFrame with fee data. Its index should be datetime.
    market_key: str,
    width: int = None,
    height: int = 1200,  # Total figure height.
) -> go.Figure:
    """
    Create a Plotly figure showing:
      1. Top subplot: ETH Price with 8h Rolling Average & Position shapes.
      2. Middle subplot: Net Value Comparison.
      3. Bottom subplot: Fees:
           - Scatter (markers) for fee per action (left Y axis).
           - Scatter (line) for cumulative fees (right Y axis).
         The bottom subplot is half as tall as the combined upper two.

    Parameters:
    - df_positions: DataFrame with position data (columns include 'open_time', 'close_time',
      'lower_quote_price', 'upper_quote_price', etc.).
    - actuator_alt: Object containing account_status_df and strategy.data for the alternative strategy.
    - actuator_np: Object containing account_status_df for the no-provision strategy.
    - df_fees: DataFrame with fee data. Its index should be datetime and it must contain:
          "fee_usd" (fee per action) and "cum_fee_usd" (cumulative fees in USD).
    - market_key: Key for accessing market-specific data from actuator_alt.strategy.data.
    - width: Figure width in pixels.
    - height: Figure height in pixels (default: 1200).

    Returns:
    - fig: A Plotly Figure object.
    """

    # Extract series from the actuators.
    eth_price_series = actuator_alt.account_status_df["price"]["ETH"]
    rolling_avg_series = actuator_alt.strategy.data[market_key]["rolling_avg"]
    net_value_series_alt = actuator_alt.account_status_df["net_value"]
    net_value_series_np = actuator_np.account_status_df["net_value"]

    # Create a figure with 3 rows.
    # Rows 1 & 2 get 0.4 each, row 3 gets 0.2 (bottom plot is half the height of the upper two combined).
    # Use specs so that row 3 has a secondary y-axis.
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            "ETH Price with 8h Rolling Average & Positions",
            "Net Value Comparison",
            "Fees: Fee per Action (markers) & Cumulative Fees (line)",
        ),
        row_heights=[0.4, 0.4, 0.2],
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": True}],
        ],
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

    # Build shapes for positions.
    shapes = []
    for _, pos in df_positions.iterrows():
        if pd.notnull(pos["close_time"]):
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
            mid_val = (pos["lower_quote_price"] + pos["upper_quote_price"]) / 2
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

    # Middle subplot: Net Value Comparison.
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

    # Bottom subplot: Fees.
    # Trace 1: Scatter markers for fee per action (left y-axis).
    fig.add_trace(
        go.Scatter(
            x=df_fees.index,
            y=df_fees["fee_usd"],
            mode="markers",
            name="Fee per Action (USD)",
            marker=dict(color="blue", size=6, opacity=0.3),
        ),
        row=3,
        col=1,
        secondary_y=False,
    )

    # Trace 2: Scatter line for cumulative fees (right y-axis).
    fig.add_trace(
        go.Scatter(
            x=df_fees.index,
            y=df_fees["cum_fee_usd"],
            mode="lines",
            name="Cumulative Fees (USD)",
            line=dict(color="gray"),
        ),
        row=3,
        col=1,
        secondary_y=True,
    )

    # Update layout: add shapes to the first subplot.
    fig.update_layout(
        title="ETH Price (with Rolling Average & Positions), Net Value, and Fees",
        template="plotly_white",
        width=width,
        height=height,
        shapes=shapes,
    )

    # Update axis titles.
    fig.update_yaxes(title_text="ETH Price (USDC)", row=1, col=1)
    fig.update_yaxes(title_text="Net Value (USDC)", row=2, col=1)
    fig.update_yaxes(title_text="Fee per Action (USD)", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Fees (USD)", row=3, col=1, secondary_y=True)

    # Update x-axis range for the bottom subplot if needed.
    fig.update_xaxes(
        range=[df_fees.index.min(), df_fees.index.max()], row=3, col=1, type="date"
    )

    return fig


def report_strategy_performance(
    actuator_test, actuator_baseline, market_key, strategy_params, strategy_results
):
    short_name = strategy_params.get_short_name()
    safe_short_name = short_name.replace(",", "dot")

    results_path = (
        f"results/{safe_short_name}-{datetime.now().isoformat(timespec='seconds')}"
    )
    os.makedirs(results_path, exist_ok=True)

    rounded_results = {
        "diff_from_baseline_at_end": round(
            strategy_results.diff_from_baseline_at_end, 2
        ),
        "percent_increase_from_baseline_at_end": round(
            strategy_results.percent_increase_from_baseline_at_end, 2
        ),
    }

    (Path(results_path) / "params.json").write_text(strategy_params.model_dump_json())
    (Path(results_path) / "results_summary.json").write_text(
        json.dumps(rounded_results, indent=2)
    )

    actuator_test.save_result(path=results_path)
    df_fees = get_fees_dataframe(actuator_test, market_key=market_key)
    df_positions = get_positions_dataframe(actuator_test._action_list)

    fig = visualise_strategy(
        df_positions, actuator_test, actuator_baseline, df_fees, market_key
    )

    html_file = os.path.join(results_path, f"{safe_short_name}.html")
    fig.write_html(html_file)

    jsonl_file = os.path.join(results_path, "action_list.jsonl")
    serialize_actions_to_jsonl(actuator_test._action_list, jsonl_file)

    return fig
